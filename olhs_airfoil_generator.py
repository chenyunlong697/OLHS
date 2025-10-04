"""Generate OLHS samples for 14D CST airfoil features and export airfoil data.

Workflow (per README.md):
    1. Create an optimized Latin hypercube (OLHS) covering the 14 design
       features extracted in ``CST.py`` for the NACA4412 baseline.
    2. For each design vector, solve an inverse CST problem so that the CST
       coefficients reproduce the target features as closely as possible.
    3. Export the resulting airfoils — feature tables, CST coefficients, and
       xFoil-ready surface points — plus a sanity-check plot of all curves.

By default the artefacts are stored in ``D:\\parameterization\\airfoil\\OLHS``.
Use ``--out-dir`` to override when running on another platform.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_mpl_spec = importlib.util.find_spec("matplotlib.pyplot")
if _mpl_spec is not None:
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None

# CST utilities live in CST.py (referenced in README.md).
from CST import (
    compute_features_14,
    cst_camber,
    cst_thickness,
    fit_camber_cst,
    fit_thickness_cst,
    naca4,
    normalize_and_align,
    resample_cosine,
)


@dataclass(frozen=True)
class FeatureSpace:
    """Bounds and helpers for the 14D feature vector."""

    names: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def to_array(self, values: Dict[str, float]) -> np.ndarray:
        return np.array([values[name] for name in self.names], dtype=float)


class OptimizedLHS:
    """Best-of-N Latin hypercube sampler with a simple correlation metric."""

    def __init__(self, n_dim: int, rng: np.random.Generator) -> None:
        self.n_dim = int(n_dim)
        self.rng = rng

    def _lhs_once(self, n_samples: int) -> np.ndarray:
        result = np.empty((n_samples, self.n_dim), dtype=float)
        for j in range(self.n_dim):
            perm = self.rng.permutation(n_samples)
            u = self.rng.random(n_samples)
            result[:, j] = (perm + u) / float(n_samples)
        return result

    def sample(self, n_samples: int, n_restarts: int = 64) -> np.ndarray:
        best = None
        best_score = math.inf
        for _ in range(max(1, n_restarts)):
            cand = self._lhs_once(n_samples)
            corr = np.corrcoef(cand, rowvar=False)
            score = float(np.sum(np.abs(corr - np.eye(self.n_dim))))
            if score < best_score:
                best_score = score
                best = cand
        if best is None:
            raise RuntimeError("Failed to build Latin hypercube")
        return best


class FeatureInverter:
    """Solve for CST coefficients that match a feature target (least-squares)."""

    def __init__(
        self,
        x_grid: np.ndarray,
        b0: Sequence[float],
        c0: Sequence[float],
        dz0: float,
        feature_space: FeatureSpace,
        rng: np.random.Generator,
        tol: float = 5e-3,
        thickness_guard: float = 8e-4,
    ) -> None:
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.b0 = np.array(b0, dtype=float)
        self.c0 = np.array(c0, dtype=float)
        self.dz0 = float(dz0)
        self.feature_space = feature_space
        self.rng = rng
        self.tol = float(tol)
        self.thickness_guard = float(thickness_guard)
        self.guard_penalty = 5.0

        self.len_b = self.b0.size
        self.len_c = self.c0.size
        self.idx_b = slice(0, self.len_b)
        self.idx_c = slice(self.len_b, self.len_b + self.len_c)
        self.idx_dz = self.len_b + self.len_c

        self.theta_init = np.concatenate([self.b0, self.c0, [self.dz0]])
        base_abs = np.maximum(np.abs(self.theta_init), 0.25)
        self.theta_lower = self.theta_init - 2.0 * base_abs
        self.theta_upper = self.theta_init + 2.0 * base_abs
        self.theta_lower[self.idx_dz] = 0.0
        self.theta_upper[self.idx_dz] = max(self.theta_upper[self.idx_dz], 0.08)

        self.fd_step = 1e-3 * np.maximum(np.abs(self.theta_init), 0.5)
        self.reg_base = 1e-4
        self.max_iter = 40
        self.theta_last = self.theta_init.copy()

    def _split_theta(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        b = theta[self.idx_b]
        c = theta[self.idx_c]
        dz = float(theta[self.idx_dz])
        return b, c, dz

    def _evaluate(self, theta: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        b, c, dz = self._split_theta(theta)
        feats, extras = compute_features_14(self.x_grid, b, c, dz)
        vec = np.array([feats[name] for name in self.feature_space.names], dtype=float)
        extras_map = {
            "t": extras["t"],
            "z": extras["z"],
            "yu": extras["z"] + 0.5 * extras["t"],
            "yl": extras["z"] - 0.5 * extras["t"],
            "t_min": float(np.min(extras["t"])),
        }
        return vec, feats, extras_map

    def solve(self, target_vec: np.ndarray, max_attempts: int = 4) -> Dict[str, object]:
        target_vec = np.array(target_vec, dtype=float)
        scale = self.feature_space.width
        best_payload: Optional[Dict[str, object]] = None
        best_err = math.inf
        init_pool = [self.theta_last.copy(), self.theta_init.copy()]

        for attempt in range(max_attempts):
            if attempt < len(init_pool):
                theta = init_pool[attempt].copy()
            else:
                jitter = 0.25 * self.rng.standard_normal(self.theta_init.shape)
                theta = self.theta_init + jitter * np.maximum(np.abs(self.theta_init), 1.0)
            theta = np.clip(theta, self.theta_lower, self.theta_upper)

            for _ in range(self.max_iter):
                vec, feats_dict, extras = self._evaluate(theta)
                residual = vec - target_vec
                res_scaled = residual / scale
                err = float(np.linalg.norm(res_scaled))
                t_min = extras["t_min"]
                guard_violation = max(0.0, self.thickness_guard - t_min)
                err_guarded = err + self.guard_penalty * guard_violation * guard_violation
                payload = {
                    "theta": theta.copy(),
                    "features": feats_dict,
                    "extras": extras,
                    "error": err,
                    "t_min": t_min,
                }
                if guard_violation <= 0.0 and err_guarded < best_err:
                    best_err = err_guarded
                    best_payload = payload
                if guard_violation <= 0.0 and err <= self.tol:
                    self.theta_last = theta.copy()
                    return payload

                jac = np.zeros((vec.size, theta.size), dtype=float)
                for j in range(theta.size):
                    step = self.fd_step[j]
                    theta_fd = theta.copy()
                    theta_fd[j] = np.clip(theta_fd[j] + step, self.theta_lower[j], self.theta_upper[j])
                    vec_fd, _, _ = self._evaluate(theta_fd)
                    jac[:, j] = (vec_fd - vec) / step
                jac_scaled = jac / scale[:, None]
                lhs = jac_scaled.T @ jac_scaled + self.reg_base * np.eye(theta.size)
                rhs = -jac_scaled.T @ res_scaled
                try:
                    delta = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError:
                    break

                success = False
                for alpha in (1.0, 0.6, 0.4, 0.2, 0.1):
                    theta_new = np.clip(theta + alpha * delta, self.theta_lower, self.theta_upper)
                    vec_new, feats_new, extras_new = self._evaluate(theta_new)
                    res_new = (vec_new - target_vec) / scale
                    err_new = float(np.linalg.norm(res_new))
                    t_min_new = extras_new["t_min"]
                    if not np.all(np.isfinite(res_new)):
                        continue
                    if t_min_new < self.thickness_guard:
                        continue
                    if err_new < err:
                        theta = theta_new
                        success = True
                        break
                if not success:
                    break

        if best_payload is None:
            raise RuntimeError("Inverse CST solver failed to converge")
        self.theta_last = best_payload["theta"].copy()
        return best_payload


def compute_baseline(n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coords = naca4()
    coords_norm = normalize_and_align(coords)
    res = resample_cosine(coords_norm, n=n_grid)
    x = res["x"]
    yu = res["yu"]
    yl = res["yl"]
    t = yu - yl
    z = 0.5 * (yu + yl)
    b, dz_te = fit_thickness_cst(x, t)
    c = fit_camber_cst(x, z)
    return x, b, c, float(dz_te)


def lhs_features(feature_space: FeatureSpace, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    sampler = OptimizedLHS(len(feature_space.names), rng)
    unit = sampler.sample(n_samples, n_restarts=96)
    return feature_space.lower + unit * feature_space.width


def build_xfoil_points(
    b: np.ndarray,
    c: np.ndarray,
    dz_te: float,
    n_points: int = 200,
) -> np.ndarray:
    if n_points < 6:
        raise ValueError("Need at least 6 points for a closed loop")
    n_upper = n_points // 2 + 1
    n_lower = n_points - n_upper + 1
    theta_upper = np.linspace(0.0, math.pi, n_upper, endpoint=True)
    theta_lower = np.linspace(0.0, math.pi, n_lower, endpoint=True)
    x_upper = 0.5 * (1.0 - np.cos(theta_upper))[::-1]
    x_lower = 0.5 * (1.0 - np.cos(theta_lower))[1:]

    def surfaces(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_clip = np.clip(x, 1e-6, 1.0 - 1e-6)
        t = cst_thickness(x_clip, b, dz_te)
        z = cst_camber(x_clip, c)
        return z + 0.5 * t, z - 0.5 * t

    yu_upper, _ = surfaces(x_upper)
    _, yl_lower = surfaces(x_lower)

    x_loop = np.concatenate([x_upper, x_lower])
    y_loop = np.concatenate([yu_upper, yl_lower])
    return np.column_stack([x_loop, y_loop])


def export_excel(
    out_path: Path,
    feature_space: FeatureSpace,
    targets: np.ndarray,
    realised: np.ndarray,
    theta_array: np.ndarray,
    theta_names: Sequence[str],
    xfoil_sets: List[np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_targets = pd.DataFrame(targets, columns=feature_space.names)
    df_realised = pd.DataFrame(realised, columns=[f"realised_{n}" for n in feature_space.names])
    df_theta = pd.DataFrame(theta_array, columns=theta_names)
    df_bundle = pd.concat([df_targets, df_realised], axis=1)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_bundle.to_excel(writer, index=False, sheet_name="design_vs_realised")
        df_theta.to_excel(writer, index=False, sheet_name="cst_coefficients")
        rows = []
        for idx, pts in enumerate(xfoil_sets):
            for seq, (x, y) in enumerate(pts):
                rows.append({"sample": idx, "sequence": seq, "x": x, "y": y})
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="xfoil_points")

    df_bundle.to_csv(out_path.with_suffix(".csv"), index=False)


def plot_airfoils(out_path: Path, x: np.ndarray, surfaces: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    if plt is None:
        print("[WARN] matplotlib is not available; skipping overlay plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.2))
    ax = plt.gca()
    if not surfaces:
        ax.text(0.5, 0.5, "No airfoils", ha="center", va="center")
    else:
        plot_count = min(len(surfaces), 40)
        sample_idx = np.linspace(0, len(surfaces) - 1, plot_count, dtype=int)
        cmap = plt.get_cmap("rainbow")
        markevery = max(1, x.size // 18)
        for rank, idx in enumerate(sample_idx):
            yu, yl = surfaces[idx]
            color = cmap(float(rank) / max(1, plot_count - 1))
            ax.plot(x, yu, color=color, lw=1.1, marker="o", markevery=markevery, markersize=2.5)
            ax.plot(x, yl, color=color, lw=1.1, marker="o", markevery=markevery, markersize=2.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("OLHS-generated CST airfoils (upper/lower do not cross)")
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate OLHS CST airfoil samples")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=400,
        help="Number of samples to generate (default: 400)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\\parameterization\\airfoil\\OLHS"),
        help="Output directory (default matches README).",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = np.random.default_rng(args.seed)

    feature_space = FeatureSpace(
        names=(
            "t_max",
            "x_t",
            "f_max",
            "x_f",
            "r_le_hat",
            "dz_te",
            "s_rec",
            "t_015",
            "t_050",
            "t_075",
            "dt_080",
            "dz_005",
            "dz_090",
            "r_fx",
        ),
        lower=np.array([0.04, 0.25, 0.10, 0.25, 0.003, 0.00, 0.01, 0.02, 0.04, 0.01, -0.60, 0.00, -0.10, -0.01], dtype=float),
        upper=np.array([0.20, 0.55, 0.12, 0.65, 0.040, 0.01, 0.06, 0.13, 0.13, 0.08, -0.05, 0.25, 0.15, 0.02], dtype=float),
    )

    print("[INFO] Fitting CST coefficients for the baseline NACA4412 airfoil...")
    x_grid, b0, c0, dz0 = compute_baseline()
    inverter = FeatureInverter(x_grid, b0, c0, dz0, feature_space, rng)
    theta_names = [f"b{i}" for i in range(inverter.len_b)] + [f"c{i}" for i in range(inverter.len_c)] + ["dz_te"]

    print(f"[INFO] Generating {args.n_samples} OLHS feature vectors...")
    targets = lhs_features(feature_space, args.n_samples, rng)

    realised = []
    theta_rows = []
    surface_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    xfoil_sets: List[np.ndarray] = []
    errors = []

    for idx, target in enumerate(targets):
        result = inverter.solve(target)
        theta = result["theta"]
        feats = result["features"]
        extras = result["extras"]
        err = float(result["error"])
        t_min = float(result["t_min"])
        errors.append(err)

        realised.append([feats[name] for name in feature_space.names])
        theta_rows.append(theta)
        surface_pairs.append((extras["yu"], extras["yl"]))

        b, c, dz = inverter._split_theta(theta)
        xfoil_sets.append(build_xfoil_points(b, c, dz))
        status = "OK" if err <= inverter.tol else "WARN"
        note = "" if status == "OK" else " <- high residual"
        print(
            f"  - sample {idx:03d} solved (residual {err:.3e}, t_min {t_min:.3e}) [{status}]"
            + note
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Writing Excel workbook and CSV summary...")
    export_excel(
        out_dir / "olhs_airfoil_samples.xlsx",
        feature_space,
        targets,
        np.array(realised),
        np.array(theta_rows),
        theta_names,
        xfoil_sets,
    )

    print("[INFO] Rendering overlay plot for the sampled airfoils...")
    plot_airfoils(out_dir / "olhs_airfoils.png", x_grid, surface_pairs)

    stats = np.array(errors)
    print(
        "[INFO] Residual norms => min {:.3e}, median {:.3e}, max {:.3e}".format(
            stats.min(), np.median(stats), stats.max()
        )
    )
    print("[INFO] Outputs saved to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
