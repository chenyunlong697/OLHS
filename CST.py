# -*- coding: utf-8 -*-
# CST.py — Airfoil CST fit + 14D feature extraction + robust plotting (EN annotations)
# Paths are hard-coded for your environment.

from pathlib import Path
import json, os, time
import numpy as np
import pandas as pd

# ===================== Config =====================
INPUT_PATH = r"D:\parameterization\airfoil\NACA4412.dat"
OUT_DIR    = Path(r"D:\parameterization\airfoil\cst_extract_out")

# sampling & CST orders
N_GRID = 200   # cosine-resampled points on [0,1]
NT     = 5     # thickness CST order
NC     = 3     # camber CST order
LAM_T  = 1e-6  # Tikhonov for thickness fit
LAM_C  = 1e-6  # Tikhonov for camber fit

# matplotlib is optional (script still runs without plots)
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
except Exception:
    HAS_MPL = False

# ===================== Utilities =====================
def cos_spacing(n: int, eps: float = 1e-4) -> np.ndarray:
    """cosine spacing in (0,1) avoiding exact 0/1 to keep class function well-conditioned."""
    k = np.arange(n)
    x = 0.5 * (1 - np.cos(np.pi * (k + 1) / (n + 1)))
    return np.clip(x, eps, 1 - eps)

def bernstein(n: int, i: int, x: np.ndarray) -> np.ndarray:
    from math import comb
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))

def bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    return np.stack([bernstein(n, i, x) for i in range(n + 1)], axis=1)

def d_bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. x of Bernstein basis of order n."""
    if n == 0:
        return np.zeros((len(x), 1))
    Bnm1 = bernstein_matrix(n - 1, x)
    Bd = np.zeros((len(x), n + 1))
    # boundary columns
    Bd[:, 0] = -n * Bnm1[:, 0]
    if n - 1 >= 1:
        Bd[:, 1:n] = n * (Bnm1[:, 0:n - 1] - Bnm1[:, 1:n])
    Bd[:, n] = n * Bnm1[:, n - 1]
    return Bd

def class_fn(x: np.ndarray) -> np.ndarray:
    """CST class function with N1=0.5, N2=1; good default for prop/airfoil."""
    return np.sqrt(x) * (1 - x)

def d_class_fn(x: np.ndarray) -> np.ndarray:
    x_safe = np.clip(x, 1e-9, 1 - 1e-9)
    return 0.5 / np.sqrt(x_safe) * (1 - x_safe) - np.sqrt(x_safe)

def interp1(x: np.ndarray, y: np.ndarray, xi) -> np.ndarray:
    return np.interp(np.atleast_1d(xi), x, y).squeeze()

def quadratic_vertex(x: np.ndarray, y: np.ndarray, i: int):
    """Parabolic fit through (i-1,i,i+1) to refine extremum location."""
    n = len(x)
    if i <= 0 or i >= n - 1:
        return float(x[i]), float(y[i])
    xi = x[i - 1:i + 2]; yi = y[i - 1:i + 2]
    A = np.stack([xi**2, xi, np.ones_like(xi)], axis=1)
    a, b, c = np.linalg.lstsq(A, yi, rcond=None)[0]
    if abs(a) < 1e-12:
        return float(x[i]), float(y[i])
    xv = -b / (2 * a)
    yv = a * xv**2 + b * xv + c
    return float(xv), float(yv)

# ===================== IO & geometry =====================
def _robust_loadtxt(path: Path):
    xs = []; ys = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t: continue
            if t[0] in "#!%(;":  # comments/titles
                continue
            parts = t.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2: continue
            try:
                xs.append(float(parts[0])); ys.append(float(parts[1]))
            except ValueError:
                continue
    if len(xs) < 5:
        return None
    return np.column_stack([np.array(xs, float), np.array(ys, float)])

def load_airfoil_any(path: str):
    """Accepts 2-column x,y in any LE-split form; splits at min(x) into upper/lower."""
    p = Path(path)
    if not p.exists():
        return None
    arr = _robust_loadtxt(p)
    if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
        return None
    x, y = arr[:, 0], arr[:, 1]
    idx_le = int(np.argmin(x))
    # upper: LE->TE ; lower: LE->TE
    xu = x[:idx_le + 1][::-1]; yu = y[:idx_le + 1][::-1]
    xl = x[idx_le:];           yl = y[idx_le:]
    return {"xu": xu, "yu": yu, "xl": xl, "yl": yl}

def normalize_and_align(coords: dict):
    """Translate LE to (0,0), rotate chord to x-axis, scale chord->1."""
    xu, yu, xl, yl = [np.array(coords[k]) for k in ("xu", "yu", "xl", "yl")]
    x_all = np.concatenate([xu, xl]); y_all = np.concatenate([yu, yl])
    le_idx = int(np.argmin(x_all))
    x0, y0 = x_all[le_idx], y_all[le_idx]
    xu, yu = xu - x0, yu - y0
    xl, yl = xl - x0, yl - y0
    teu = int(np.argmax(xu)); tel = int(np.argmax(xl))
    te_xy = 0.5 * (np.array([xu[teu], yu[teu]]) + np.array([xl[tel], yl[tel]]))
    theta = np.arctan2(te_xy[1], te_xy[0])
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta),  np.cos(-theta)]])
    U = (R @ np.vstack([xu, yu])).T
    L = (R @ np.vstack([xl, yl])).T
    xu, yu = U[:, 0], U[:, 1]; xl, yl = L[:, 0], L[:, 1]
    chord = max(float(np.max(xu)), float(np.max(xl)))
    if chord <= 0: chord = 1.0
    xu, yu = xu / chord, yu / chord
    xl, yl = xl / chord, yl / chord
    return {"xu": xu, "yu": yu, "xl": xl, "yl": yl, "chord": chord}

def resample_cosine(coords: dict, n: int = 200):
    xg = cos_spacing(n)
    def sort_unique(x, y):
        idx = np.argsort(x); x = x[idx]; y = y[idx]
        ux, iux = np.unique(x, return_index=True)
        return ux, y[iux]
    xu, yu = sort_unique(coords["xu"], coords["yu"])
    xl, yl = sort_unique(coords["xl"], coords["yl"])
    yu_i = np.interp(xg, xu, yu)
    yl_i = np.interp(xg, xl, yl)
    return {"x": xg, "yu": yu_i, "yl": yl_i}

def naca4(m=0.04, p=0.4, t=0.12, n=1201):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    return {"xu": xu[::-1], "yu": yu[::-1], "xl": xl[1:], "yl": yl[1:]}

# ===================== CST model =====================
def fit_thickness_cst(x, t, nt=NT, lam=LAM_T):
    C = class_fn(x)[:, None]; B = bernstein_matrix(nt, x)
    A = np.hstack([C * B, x[:, None]])  # last column represents x*dz_te
    theta = np.linalg.lstsq(
        np.vstack([A, np.sqrt(lam)*np.eye(A.shape[1])]),
        np.hstack([t, np.zeros(A.shape[1])]),
        rcond=None
    )[0]
    b = theta[:-1]; dz_te = theta[-1]
    return b, float(dz_te)

def fit_camber_cst(x, z, nc=NC, lam=LAM_C):
    C = class_fn(x)[:, None]; B = bernstein_matrix(nc, x)
    A = C * B
    c = np.linalg.lstsq(
        np.vstack([A, np.sqrt(lam)*np.eye(A.shape[1])]),
        np.hstack([z, np.zeros(A.shape[1])]),
        rcond=None
    )[0]
    return c

def cst_thickness(x, b, dz_te):
    C = class_fn(x); B = bernstein_matrix(len(b)-1, x)
    return C * (B @ b) + x * dz_te

def cst_camber(x, c):
    C = class_fn(x); B = bernstein_matrix(len(c)-1, x)
    return C * (B @ c)

def cst_thickness_deriv(x, b, dz_te):
    C = class_fn(x); Cp = d_class_fn(x)
    B = bernstein_matrix(len(b)-1, x); Bd = d_bernstein_matrix(len(b)-1, x)
    return Cp * (B @ b) + C * (Bd @ b) + dz_te

def cst_camber_deriv(x, c):
    C = class_fn(x); Cp = d_class_fn(x)
    B = bernstein_matrix(len(c)-1, x); Bd = d_bernstein_matrix(len(c)-1, x)
    return Cp * (B @ c) + C * (Bd @ c)

# ===================== 14D features =====================
def compute_features_14(x, b, c, dz_te):
    t = cst_thickness(x, b, dz_te)
    z = cst_camber(x, c)
    tp = cst_thickness_deriv(x, b, dz_te)
    zp = cst_camber_deriv(x, c)

    i_tmax = int(np.argmax(t)); x_t, t_max = quadratic_vertex(x, t, i_tmax)
    i_fmax = int(np.argmax(np.abs(z))); x_f, f_max = quadratic_vertex(x, z, i_fmax)

    # LE radius proxy from t/2 ~ a sqrt(x) + b x in [0.003, 0.03]
    xfit = np.array([0.003, 0.007, 0.015, 0.03])
    yfit = interp1(x, 0.5*t, xfit)
    A = np.stack([np.sqrt(xfit), xfit], axis=1)
    a, _b = np.linalg.lstsq(A, yfit, rcond=None)[0]
    r_le_hat = float((a*a)/2.0)

    dz_te_eff = float(interp1(x, t, 1.0 - 1e-6))
    s_rec = float(interp1(x, t, 0.6) - interp1(x, t, 0.9))
    t_015 = float(interp1(x, t, 0.15))
    t_050 = float(interp1(x, t, 0.50))
    t_075 = float(interp1(x, t, 0.75))
    dt_080 = float(interp1(x, tp, 0.80))
    dz_005 = float(interp1(x, zp, 0.05))
    dz_090 = float(interp1(x, zp, 0.90))
    r_fx = float(interp1(x, z, 0.90) - (interp1(x, z, 0.70) + 0.20*(interp1(x, z, 0.95) - interp1(x, z, 0.70))))

    features = {
        "t_max": float(t_max), "x_t": float(x_t),
        "f_max": float(f_max), "x_f": float(x_f),
        "r_le_hat": float(r_le_hat),
        "dz_te": dz_te_eff,
        "s_rec": float(s_rec),
        "t_015": t_015, "t_050": t_050, "t_075": t_075,
        "dt_080": dt_080, "dz_005": dz_005, "dz_090": dz_090,
        "r_fx": r_fx
    }
    return features, {"t": t, "z": z, "tp": tp, "zp": zp}

# ===================== Robust saving =====================
def save_csv_robust(df, path: Path, retries: int = 4, sleep_s: float = 0.6) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            os.chmod(path, 0o666)
    except Exception:
        pass
    for _ in range(retries):
        try:
            df.to_csv(path, index=False, encoding="utf-8")
            return path
        except PermissionError:
            time.sleep(sleep_s)
    ts = time.strftime("%Y%m%d_%H%M%S")
    alt = path.with_name(f"{path.stem}_{ts}.csv")
    df.to_csv(alt, index=False, encoding="utf-8")
    print(f"[WARN] '{path.name}' locked. Saved as '{alt.name}'.")
    return alt

def save_json_robust(obj, path: Path, retries: int = 4, sleep_s: float = 0.6) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            os.chmod(path, 0o666)
    except Exception:
        pass
    for _ in range(retries):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            return path
        except PermissionError:
            time.sleep(sleep_s)
    ts = time.strftime("%Y%m%d_%H%M%S")
    alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
    with open(alt, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[WARN] '{path.name}' locked. Saved as '{alt.name}'.")
    return alt

# ===================== Plotting =====================
def plot_cst_fit_points(x, yu, yl, yu_fit, yl_fit, out_path: Path):
    if not HAS_MPL:
        print("[INFO] matplotlib unavailable; skip:", out_path.name); return
    plt.figure(figsize=(10.8, 4.2))
    ax = plt.gca()
    ax.plot(x, yu, label="Upper (orig)")
    ax.plot(x, yl, label="Lower (orig)")
    ax.plot(x, yu_fit, ls="None", marker="o", ms=3, label="Upper (CST pts)")
    ax.plot(x, yl_fit, ls="None", marker="o", ms=3, label="Lower (CST pts)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title("CST fit — points showing cosine spacing", pad=12)
    ax.legend(ncol=2, fontsize=9)


def plot_points_zoom(x, yu_fit, yl_fit, out_path: Path, xlim=(0.0, 0.12)):
    if not HAS_MPL: return
    plt.figure(figsize=(7.5, 4.2))
    ax = plt.gca()
    ax.plot(x, yu_fit, ls="None", marker="o", ms=3, label="Upper (CST pts)")
    ax.plot(x, yl_fit, ls="None", marker="o", ms=3, label="Lower (CST pts)")
    ax.set_xlim(*xlim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title("LE zoom — cosine clustering", pad=10)
    ax.legend()


def plot_annotated_14D_EN(x, yu, yl, z, t, out_path: Path):
    """
    14D 注释：每个竖线标签与虚线严格同 x，且位于坐标框外侧；区域大标签保留。
    """
    if not HAS_MPL:
        print("[INFO] matplotlib unavailable; skip:", out_path.name); return

    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    # ---------- Figure / Axes ----------
    fig = plt.figure(figsize=(14.4, 6.2))
    ax  = plt.gca()

    yu_fit = z + 0.5*t
    yl_fit = z - 0.5*t

    # ---------- Background spans ----------
    ax.axvspan(0.003, 0.030, facecolor="#bcdcff", alpha=0.55,
               edgecolor="#2f5b9a", linewidth=1.2, zorder=0.10)            # LE
    ax.axvspan(0.60,  0.90,  facecolor="#cfe8ff", alpha=0.35,
               edgecolor="#6fa7d8", linewidth=0.8,  zorder=0.12)            # Pressure-recovery
    ax.axvspan(0.70,  0.95,  facecolor="#e6f0ff", alpha=0.22,
               edgecolor="#385f99", linewidth=1.0,  hatch="////", zorder=0.14)  # Reflex

    # ---------- Curves ----------
    ax.plot(x, yu,     label="Upper (orig)",                    zorder=1.6)
    ax.plot(x, yl,     label="Lower (orig)",                    zorder=1.6)
    ax.plot(x, yu_fit, ls="None", marker="o", ms=3, label="Upper (CST pts)", zorder=2.0)
    ax.plot(x, yl_fit, ls="None", marker="o", ms=3, label="Lower (CST pts)", zorder=2.0)
    ax.plot(x, z,      ls=":",            label="Camber (CST)",             zorder=1.6)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title("CST fit with 14D feature annotations", pad=26)

    # ！！！给轴外文字留位置（不要再 tight_layout / bbox_inches="tight"）
    fig.subplots_adjust(left=0.09, right=0.70, bottom=0.34, top=0.82)

    # ---------- Limits ----------
    ymax_curve = float(max(np.max(yu), np.max(yu_fit)))
    ymin_curve = float(min(np.min(yl), np.min(yl_fit)))
    y_range    = max(1e-6, ymax_curve - ymin_curve)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(ymin_curve - 0.70*y_range, ymax_curve + 0.95*y_range)

    # ---------- Helpers (vlines + outside labels) ----------
    def vline(xv): ax.axvline(xv, ls=":", lw=0.9, zorder=1.2)

    # x 用数据坐标，y 用轴分数；>1 在上外侧，<0 在下外侧
    trans_data_ax = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    Y_OUT_TOP, Y_OUT_BOT = 1.08, -0.16  # 可按需要微调
    TXT_KW = dict(rotation=90, ha="center",
                  fontsize=9, color="#222",
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.6),
                  transform=trans_data_ax, clip_on=False, zorder=3.2)

    def label_top(xv, text):
        ax.annotate(text, (xv, Y_OUT_TOP), xytext=(0, 0),
                    textcoords="offset points", va="bottom", **TXT_KW)

    def label_bot(xv, text):
        ax.annotate(text, (xv, Y_OUT_BOT), xytext=(0, 0),
                    textcoords="offset points", va="top", **TXT_KW)

    # ---------- Region labels（保持不变） ----------
    def region_label(x0, x1, text, y_frac=0.82, color="#1f3f78"):
        xm = 0.5*(x0 + x1)
        ax.text(xm, y_frac, text, transform=trans_data_ax,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=color, zorder=3.2)

    region_label(0.003, 0.030, "LE radius samples (0.003–0.03)",   y_frac=0.88, color="#2f5b9a")
    region_label(0.60,  0.90,  "Pressure-recovery samples (0.60, 0.90)", y_frac=0.18, color="#2b5d94")
    region_label(0.70,  0.95,  "Reflex samples (0.70, 0.90, 0.95)", y_frac=0.88, color="#385f99")

    # ---------- 14D 单点类标签 ----------
    i_tmax = int(np.argmax(t)); x_t, _ = quadratic_vertex(x, t, i_tmax)
    i_fmax = int(np.argmax(np.abs(z))); x_f, _ = quadratic_vertex(x, z, i_fmax)

    singles = [
        ("top",    x_t,  "Max thickness"),
        ("bottom", x_f,  "Max camber"),
        ("bottom", 0.15, "t(0.15)"),
        ("top",    0.50, "t(0.50)"),
        ("bottom", 0.75, "t(0.75)"),
        ("top",    0.80, "t'(0.80)"),
        ("bottom", 0.05, "z'(0.05)"),
        ("top",    0.90, "z'(0.90)"),
        ("bottom", 1.00, "TE thickness"),
    ]
    # 严格同 x：不再做水平位移
    for side, xv, text in sorted(singles, key=lambda s: s[1]):
        vline(xv)
        (label_top if side == "top" else label_bot)(xv, text)

    # 极值点
    ax.scatter([x_t], [np.interp(x_t, x, z + 0.5*t)], s=26, zorder=3.1)
    ax.scatter([x_f], [np.interp(x_f, x, z)],         s=26, marker='s', zorder=3.1)

    # Legend 右侧
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
              framealpha=0.95, fontsize=9)

    # 切记不要 tight_layout / bbox_inches="tight"
    plt.savefig(out_path, dpi=180)
    plt.close()



# ===================== Main =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coords_raw = load_airfoil_any(INPUT_PATH)
    used_fallback = False
    if coords_raw is None:
        print(f"[WARN] Cannot read '{INPUT_PATH}'. Falling back to NACA4412-like sample.")
        used_fallback = True
        coords_raw = naca4()

    coords_norm = normalize_and_align(coords_raw)
    res = resample_cosine(coords_norm, n=N_GRID)
    xg = res["x"]; yu = res["yu"]; yl = res["yl"]

    # reference thickness/camber from resampled original
    t_sample = yu - yl
    z_sample = 0.5 * (yu + yl)

    # CST fits
    b, dz_te = fit_thickness_cst(xg, t_sample, nt=NT, lam=LAM_T)
    c = fit_camber_cst(xg, z_sample, nc=NC, lam=LAM_C)

    # recon
    t_fit = cst_thickness(xg, b, dz_te)
    z_fit = cst_camber(xg, c)
    yu_fit = z_fit + 0.5 * t_fit
    yl_fit = z_fit - 0.5 * t_fit

    # features
    features14, _ = compute_features_14(xg, b, c, dz_te)

    # exports (robust save)
    df_out = pd.DataFrame({
        "x": xg, "yu": yu, "yl": yl,
        "yu_fit": yu_fit, "yl_fit": yl_fit,
        "t": t_sample, "t_fit": t_fit,
        "z": z_sample, "z_fit": z_fit
    })
    csv_path = save_csv_robust(df_out, OUT_DIR / "coords_resampled.csv")

    _ = save_json_robust({"b": list(map(float, b)),
                          "c": list(map(float, c)),
                          "dz_te": float(dz_te)},
                         OUT_DIR / "cst_coeffs.json")

    _ = save_json_robust({"features_14": features14,
                          "meta": {"input_path": INPUT_PATH,
                                   "used_fallback": used_fallback,
                                   "nt": NT, "nc": NC, "n_grid": N_GRID}},
                         OUT_DIR / "features.json")

    # plots
    plot_cst_fit_points(xg, yu, yl, yu_fit, yl_fit, OUT_DIR / "cst_fit_points.png")
    plot_annotated_14D_EN(xg, yu, yl, z_fit, t_fit, OUT_DIR / "cst_fit_preview_annotated.png")
    plot_points_zoom(xg, yu_fit, yl_fit, OUT_DIR / "cst_fit_points_zoom.png", xlim=(0.0, 0.12))

    print("[DONE] Outputs folder:", OUT_DIR)
    print(" - CSV:", csv_path.name)
    print(" - JSON:", "cst_coeffs.json, features.json")
    print(" - PNG :", "cst_fit_points.png, cst_fit_preview_annotated.png, cst_fit_points_zoom.png")

if __name__ == "__main__":
    main()
