"""Generate OLHS-based airfoil sample space with CST reconstruction.

This script follows the requirements from the project README:

* Build an Optimal Latin Hypercube Sampling (OLHS) design with 200 points
  across the 14 geometric design variables extracted from the CST fit.
* Respect the additional ratio/ordering constraints among the thickness
  features during sampling.
* For each sampled design vector, solve for CST thickness/camber coefficients
  such that the 14 features are matched while enforcing positive thickness
  across the chord.
* Export three Excel workbooks that separately capture (1) the 14D feature
  values, (2) the CST coefficients, and (3) cosine-clustered surface points
  in the XFoil-friendly ordering (upper TE -> LE -> lower TE) for all airfoils.
* Plot 10 representative airfoils (selected via farthest-point sampling in the
  normalised feature space) using distinct colours per airfoil while keeping
  upper/lower surfaces of the same airfoil identically coloured.

The script is designed for "one-click" execution and writes all artefacts into
the configured output directory (defaulting to the Windows path requested in
the README, with a sensible fallback when executed on non-Windows systems).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import CST

RAW_OUT_DIR = r"D:\\parameterization\\airfoil\\OLHS"


def resolve_output_dir(raw_path: str) -> Path:
    """Return an OS-appropriate output directory for the script."""

    if os.name == "nt":
        return Path(raw_path)

    sanitised = raw_path.replace(":", "").replace("\\", "/")
    return Path(sanitised)


OUT_DIR = resolve_output_dir(RAW_OUT_DIR)

N_SAMPLES = 200
N_DIM = 14
N_POINTS_PER_AIRFOIL = 200

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "t_max": (0.04, 0.20),
    "x_t": (0.25, 0.55),
    "f_max": (0.10, 0.12),
    "x_f": (0.40, 0.55),
    "r_le_hat": (0.01, 0.018),
    "dz_te": (0.0, 0.006),
    "s_rec": (0.015, 0.045),
    "t_015": (0.035, 0.080),
    "t_050": (0.040, 0.130),
    "t_075": (0.010, 0.080),
    "dt_080": (-0.30, -0.07),
    "dz_005": (0.08, 0.16),
    "dz_090": (-0.10, 0.15),
    "r_fx": (-0.005, 0.012),
}

FEATURE_ORDER: List[str] = list(PARAM_BOUNDS.keys())
FEATURE_WEIGHTS = np.array([
    1.0 / (PARAM_BOUNDS[name][1] - PARAM_BOUNDS[name][0]) for name in FEATURE_ORDER
])


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling utilities
# ---------------------------------------------------------------------------


def lhs_random(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Construct a single random Latin Hypercube in [0, 1]^d."""

    result = np.empty((n_samples, n_dim), dtype=float)
    interval = np.linspace(0.0, 1.0, n_samples + 1)
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        low = interval[perm]
        high = interval[perm + 1]
        result[:, j] = rng.uniform(low, high)
    return result


def pairwise_min_distance(samples: np.ndarray) -> float:
    """Return the minimum pairwise Euclidean distance for the given samples."""

    diff = samples[:, None, :] - samples[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    return float(np.sqrt(np.min(dist_sq)))


def olhs(n_samples: int, n_dim: int, iterations: int = 200, seed: int = 2024) -> np.ndarray:
    """Generate an OLHS design via maximin search over random LH samples."""

    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    for _ in range(iterations):
        candidate = lhs_random(n_samples, n_dim, rng)
        score = pairwise_min_distance(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Feature sampling with inequality constraints
# ---------------------------------------------------------------------------


def scale_to_bounds(u: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return lo + (hi - lo) * u


def map_unit_to_features(unit_sample: Sequence[float]) -> Dict[str, float]:
    """Map a unit-cube sample to the 14D feature vector respecting constraints."""

    u = list(unit_sample)
    features: Dict[str, float] = {}

    features["t_max"] = scale_to_bounds(u[0], PARAM_BOUNDS["t_max"])
    features["x_t"] = scale_to_bounds(u[1], PARAM_BOUNDS["x_t"])
    features["f_max"] = scale_to_bounds(u[2], PARAM_BOUNDS["f_max"])
    features["x_f"] = scale_to_bounds(u[3], PARAM_BOUNDS["x_f"])
    features["r_le_hat"] = scale_to_bounds(u[4], PARAM_BOUNDS["r_le_hat"])
    features["dz_te"] = scale_to_bounds(u[5], PARAM_BOUNDS["dz_te"])
    features["s_rec"] = scale_to_bounds(u[6], PARAM_BOUNDS["s_rec"])

    t_max = features["t_max"]

    t015_lo = max(PARAM_BOUNDS["t_015"][0], 0.35 * t_max)
    t015_hi = min(PARAM_BOUNDS["t_015"][1], 0.95 * t_max)
    if t015_hi <= t015_lo:
        t015_hi = t015_lo + 1e-5
    features["t_015"] = t015_lo + (t015_hi - t015_lo) * u[7]

    t050_hi = min(PARAM_BOUNDS["t_050"][1], t_max)
    t050_lo = PARAM_BOUNDS["t_050"][0]
    if t050_hi <= t050_lo:
        features["t_050"] = t050_hi
    else:
        features["t_050"] = t050_lo + (t050_hi - t050_lo) * u[8]

    t075_hi = min(PARAM_BOUNDS["t_075"][1], features["t_050"])
    t075_lo = PARAM_BOUNDS["t_075"][0]
    if t075_hi <= t075_lo:
        features["t_075"] = t075_hi
    else:
        features["t_075"] = t075_lo + (t075_hi - t075_lo) * u[9]

    features["dt_080"] = scale_to_bounds(u[10], PARAM_BOUNDS["dt_080"])
    features["dz_005"] = scale_to_bounds(u[11], PARAM_BOUNDS["dz_005"])
    features["dz_090"] = scale_to_bounds(u[12], PARAM_BOUNDS["dz_090"])
    features["r_fx"] = scale_to_bounds(u[13], PARAM_BOUNDS["r_fx"])

    return features


# ---------------------------------------------------------------------------
# CST reconstruction utilities
# ---------------------------------------------------------------------------


def baseline_theta_and_features() -> Tuple[np.ndarray, np.ndarray]:
    """Fit CST coefficients to the reference airfoil and obtain features."""

    coords_raw = CST.load_airfoil_any(CST.INPUT_PATH)
    if coords_raw is None:
        coords_raw = CST.naca4()

    coords_norm = CST.normalize_and_align(coords_raw)
    res = CST.resample_cosine(coords_norm, n=CST.N_GRID)
    x = res["x"]
    yu = res["yu"]
    yl = res["yl"]

    thickness = yu - yl
    camber = 0.5 * (yu + yl)

    b, dz_te = CST.fit_thickness_cst(x, thickness, nt=CST.NT, lam=CST.LAM_T)
    c = CST.fit_camber_cst(x, camber, nc=CST.NC, lam=CST.LAM_C)

    theta = np.concatenate([b, c, [dz_te]])
    x_dense = CST.cos_spacing(400)
    feats, _ = CST.compute_features_14(x_dense, b, c, dz_te)
    vec = np.array([feats[name] for name in FEATURE_ORDER])
    return theta, vec


X_DENSE = CST.cos_spacing(400)


def unpack_theta(theta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split the optimisation vector into thickness, camber, and trailing edge."""

    theta = np.asarray(theta, dtype=float)
    b = theta[: CST.NT + 1]
    c = theta[CST.NT + 1 : CST.NT + 1 + CST.NC + 1]
    dz_te = float(theta[-1])
    return b, c, dz_te


def leading_edge_monotone(thickness: np.ndarray, x: np.ndarray, tol: float = 2.5e-4) -> bool:
    """Return True when thickness grows monotonically from the leading edge."""

    idx = int(np.searchsorted(x, 0.22))
    if idx <= 2:
        return True
    diffs = np.diff(thickness[: idx + 1])
    return bool(np.all(diffs >= -tol))


def leading_edge_camber_rises(camber: np.ndarray, x: np.ndarray, tol: float = 3.0e-4) -> bool:
    """Ensure the camber line does not dip below the leading-edge height."""

    idx = int(np.searchsorted(x, 0.08))
    if idx <= 2:
        return True
    baseline = camber[0]
    window = camber[: idx + 1] - baseline
    return bool(np.all(window >= -tol))


def leading_edge_thickness_sufficient(
    thickness: np.ndarray,
    x: np.ndarray,
    min_abs: float = 0.012,
    min_ratio: float = 0.42,
) -> bool:
    """Guarantee a stout, convex nose by enforcing early-chord thickness."""

    t_015 = float(np.interp(0.15, x, thickness))
    target = max(min_abs, min_ratio * t_015)
    t_020 = float(np.interp(0.02, x, thickness))
    t_040 = float(np.interp(0.04, x, thickness))
    return bool(t_020 >= 0.65 * target and t_040 >= target)


def evaluate_theta(theta: Sequence[float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return the feature vector and auxiliary fields for a given theta."""

    b, c, dz_te = unpack_theta(theta)
    t = CST.cst_thickness(X_DENSE, b, dz_te)
    z = CST.cst_camber(X_DENSE, c)
    feats, aux = CST.compute_features_14(X_DENSE, b, c, dz_te)
    vec = np.array([feats[name] for name in FEATURE_ORDER])
    aux = dict(aux)
    aux["t"] = t
    aux["z"] = z
    aux["yu"] = z + 0.5 * t
    aux["yl"] = z - 0.5 * t
    aux["leading_edge_ok"] = leading_edge_monotone(t, X_DENSE)
    aux["leading_edge_camber_ok"] = leading_edge_camber_rises(z, X_DENSE)
    aux["leading_edge_thickness_ok"] = leading_edge_thickness_sufficient(t, X_DENSE)
    return vec, aux


def approx_jacobian(theta: np.ndarray, base_vec: np.ndarray, step_scale: float = 1e-4) -> np.ndarray:
    """Finite-difference Jacobian of the feature vector w.r.t theta."""

    theta = np.asarray(theta, dtype=float)
    n_params = theta.size
    n_features = base_vec.size
    jac = np.empty((n_features, n_params), dtype=float)

    for i in range(n_params):
        step = step_scale * max(1.0, abs(theta[i]))
        theta_eps = theta.copy()
        theta_eps[i] += step
        vec_eps, _ = evaluate_theta(theta_eps)
        jac[:, i] = (vec_eps - base_vec) / step
    return jac


@dataclass
class SolveResult:
    theta: np.ndarray
    vec: np.ndarray
    aux: Dict[str, np.ndarray]
    converged: bool
    iterations: int


def solve_for_theta(
    target_vec: np.ndarray,
    theta_init: np.ndarray,
    theta_ref: np.ndarray,
    max_iter: int = 35,
    tol: float = 1.5e-3,
    reg_strength: float = 0.015,
) -> SolveResult:
    """Solve for CST coefficients that match the requested feature vector."""

    theta = theta_init.copy()
    best_theta = theta.copy()
    best_vec, best_aux = evaluate_theta(theta)
    best_residual = (best_vec - target_vec) * FEATURE_WEIGHTS
    best_norm = float(np.linalg.norm(best_residual))

    damping = 1e-3

    for iteration in range(1, max_iter + 1):
        vec, aux = evaluate_theta(theta)
        residual = (vec - target_vec) * FEATURE_WEIGHTS
        norm = float(np.linalg.norm(residual))

        if norm < best_norm:
            best_norm = norm
            best_theta = theta.copy()
            best_vec = vec
            best_aux = aux

        if norm < tol:
            return SolveResult(theta=theta, vec=vec, aux=aux, converged=True, iterations=iteration)

        jac = approx_jacobian(theta, vec)
        jac_w = jac * FEATURE_WEIGHTS[:, None]

        reg = reg_strength ** 2
        lhs = jac_w.T @ jac_w + (damping + reg) * np.eye(theta.size)
        rhs = -jac_w.T @ residual - reg * (theta - theta_ref)

        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        success = False
        for step_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            theta_trial = theta + step_scale * delta
            vec_trial, aux_trial = evaluate_theta(theta_trial)
            residual_trial = (vec_trial - target_vec) * FEATURE_WEIGHTS
            if np.min(aux_trial["t"]) <= 5e-4:
                continue
            if not aux_trial.get("leading_edge_ok", True):
                continue
            if not aux_trial.get("leading_edge_camber_ok", True):
                continue
            if not aux_trial.get("leading_edge_thickness_ok", True):
                continue
            if np.linalg.norm(residual_trial) <= norm:
                theta = theta_trial
                damping = max(damping * 0.7, 1e-5)
                success = True
                break
        if not success:
            damping *= 5.0

    return SolveResult(theta=best_theta, vec=best_vec, aux=best_aux, converged=False, iterations=max_iter)


# ---------------------------------------------------------------------------
# Cosine-clustered surface reconstruction
# ---------------------------------------------------------------------------


def cosine_surface_points(
    x: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
    n_points: int = N_POINTS_PER_AIRFOIL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cosine-clustered upper/lower surface coordinates."""

    n_upper = n_points // 2 + 1
    n_lower = n_points - n_upper + 1

    beta_upper = np.linspace(0.0, math.pi, n_upper)
    beta_lower = np.linspace(0.0, math.pi, n_lower)

    x_upper = 0.5 * (1.0 - np.cos(beta_upper))[::-1]
    x_lower = 0.5 * (1.0 - np.cos(beta_lower))[1:]

    yu_interp = np.interp(x_upper, x, yu)
    yl_interp = np.interp(x_lower, x, yl)

    return x_upper, yu_interp, x_lower, yl_interp


def assemble_airfoil_path(
    x_upper: np.ndarray,
    yu_upper: np.ndarray,
    x_lower: np.ndarray,
    yl_lower: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_path = np.concatenate([x_upper, x_lower])
    y_path = np.concatenate([yu_upper, yl_lower])
    return x_path, y_path


# ---------------------------------------------------------------------------
# Representative airfoil selection
# ---------------------------------------------------------------------------


def farthest_point_indices(data: np.ndarray, k: int) -> List[int]:
    """Select k representative indices via farthest-point sampling."""

    if k >= len(data):
        return list(range(len(data)))

    mean = np.mean(data, axis=0)
    first = int(np.argmax(np.linalg.norm(data - mean, axis=1)))
    selected = [first]

    distances = np.linalg.norm(data - data[first], axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(distances))
        selected.append(idx)
        new_dist = np.linalg.norm(data - data[idx], axis=1)
        distances = np.minimum(distances, new_dist)
    return selected


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_dataframes(
    feature_rows: List[Dict[str, float]],
    solve_results: List[SolveResult],
    path_records: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = pd.DataFrame(feature_rows)

    coeff_rows = []
    for idx, res in enumerate(solve_results):
        b, c, dz_te = unpack_theta(res.theta)
        coeff_rows.append(
            {
                "sample_id": idx + 1,
                **{f"b{i}": val for i, val in enumerate(b)},
                "dz_te": dz_te,
                **{f"c{i}": val for i, val in enumerate(c)},
                "converged": res.converged,
                "iterations": res.iterations,
            }
        )

    coeff_df = pd.DataFrame(coeff_rows)
    points_df = pd.DataFrame(path_records)

    return feature_df, coeff_df, points_df


def save_excel(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_theta, _ = baseline_theta_and_features()
    lhs_unit = olhs(N_SAMPLES, N_DIM, iterations=250)

    feature_rows = []
    solve_results: List[SolveResult] = []
    path_records: List[Dict[str, object]] = []
    geometries = []
    theta_seed = base_theta.copy()

    for idx, u in enumerate(lhs_unit):
        features = map_unit_to_features(u)
        target_vec = np.array([features[name] for name in FEATURE_ORDER])
        solve = solve_for_theta(target_vec, theta_seed.copy(), base_theta.copy())
        solve_results.append(solve)

        theta_seed = solve.theta.copy()

        if not solve.aux.get("leading_edge_ok", True):
            raise RuntimeError("Leading-edge monotonicity violated after solve")
        if not solve.aux.get("leading_edge_camber_ok", True):
            raise RuntimeError("Leading-edge camber dips below nose height")
        if not solve.aux.get("leading_edge_thickness_ok", True):
            raise RuntimeError("Leading-edge thickness too small for convex nose")

        actual_features = {name: solve.vec[i] for i, name in enumerate(FEATURE_ORDER)}
        row = {"sample_id": idx + 1}
        for name in FEATURE_ORDER:
            row[f"{name}_target"] = features[name]
            row[f"{name}_actual"] = actual_features[name]
        row["converged"] = solve.converged
        row["iterations"] = solve.iterations
        feature_rows.append(row)

        x_upper, yu_upper, x_lower, yl_lower = cosine_surface_points(
            X_DENSE, solve.aux["yu"], solve.aux["yl"], N_POINTS_PER_AIRFOIL
        )
        x_path, y_path = assemble_airfoil_path(x_upper, yu_upper, x_lower, yl_lower)

        geometries.append(
            {
                "x_upper": x_upper,
                "y_upper": yu_upper,
                "x_lower": x_lower,
                "y_lower": yl_lower,
            }
        )

        upper_len = len(x_upper)
        for i, (x_val, y_val) in enumerate(zip(x_path, y_path), start=1):
            path_records.append(
                {
                    "sample_id": idx + 1,
                    "point_index": i,
                    "surface": "upper" if i <= upper_len else "lower",
                    "x": x_val,
                    "y": y_val,
                }
            )

    feature_df, coeff_df, points_df = build_dataframes(feature_rows, solve_results, path_records)

    save_excel(feature_df, OUT_DIR / "airfoil_features.xlsx", sheet_name="features")
    save_excel(coeff_df, OUT_DIR / "cst_coefficients.xlsx", sheet_name="coefficients")
    save_excel(points_df, OUT_DIR / "airfoil_points.xlsx", sheet_name="points")

    actual_cols = [f"{name}_actual" for name in FEATURE_ORDER]
    normalised_features = feature_df[["sample_id"] + actual_cols].copy()
    feature_array = normalised_features[actual_cols].to_numpy()
    bounds_lo = np.array([PARAM_BOUNDS[name][0] for name in FEATURE_ORDER])
    bounds_hi = np.array([PARAM_BOUNDS[name][1] for name in FEATURE_ORDER])
    feature_scaled = (feature_array - bounds_lo) / (bounds_hi - bounds_lo)

    rep_indices = farthest_point_indices(feature_scaled, 10)

    colours = plt.cm.tab10(np.linspace(0.0, 1.0, len(rep_indices)))
    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()

    for colour, idx in zip(colours, rep_indices):
        geom = geometries[idx]
        label = f"Sample {idx + 1}"
        ax.plot(geom["x_upper"], geom["y_upper"], color=colour, label=label)
        ax.plot(geom["x_lower"], geom["y_lower"], color=colour)

    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Representative airfoils from OLHS design")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "representative_airfoils.png", dpi=220)
    plt.close()

    summary = {
        "output_dir": str(OUT_DIR.resolve()),
        "feature_file": str((OUT_DIR / "airfoil_features.xlsx").resolve()),
        "coeff_file": str((OUT_DIR / "cst_coefficients.xlsx").resolve()),
        "points_file": str((OUT_DIR / "airfoil_points.xlsx").resolve()),
        "plot_file": str((OUT_DIR / "representative_airfoils.png").resolve()),
    }

    for key, value in summary.items():
        print(f"{key:>15}: {value}")


if __name__ == "__main__":
    main()
