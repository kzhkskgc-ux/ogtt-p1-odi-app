import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


# ------------------------------------------------------------
# Time grids
# ------------------------------------------------------------
T_OBS = np.array([0, 30, 60, 90, 120], dtype=float)
T_FIT = T_OBS.copy()
DENSE_T = np.linspace(0.0, 120.0, 241)

BG_COLS = ["O-BG(0)", "O-BG(30)", "O-BG(60)", "O-BG(90)", "O-BG(120)"]
IRI_COLS = ["O-IRI(0)", "O-IRI(30)", "O-IRI(60)", "O-IRI(90)", "O-IRI(120)"]
REQUIRED_COLS = ["ID"] + BG_COLS + IRI_COLS


# ------------------------------------------------------------
# Model constants
# ------------------------------------------------------------
# p2 is fixed to improve identifiability when only 5 OGTT time points are available.
# p1 is still estimated, but within a physiologically plausible range.
P2_FIXED = 0.03       # min^-1
P1_PRIOR = 0.016      # min^-1, weak prior center
GLUCOSE_SD = 10.0     # mg/dL, residual scaling


@dataclass
class FitSummary:
    metrics: Dict[str, float]
    fit_table: pd.DataFrame
    dense_table: pd.DataFrame


def simulate_p1_si_model(
    t_eval: np.ndarray,
    t_ins: np.ndarray,
    ins_uU: np.ndarray,
    Gb: float,
    Ib: float,
    p1: float,
    SI: float,
    KGI: float,
    Ka: float,
    GI0: float,
    p2_fixed: float = P2_FIXED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified oral minimal model.

    G  : plasma glucose concentration, mg/dL
    X  : remote insulin action
    GI : gastrointestinal glucose compartment
    E  : enterocyte / appearance precursor compartment
    Ra : model-derived glucose appearance index, mg/dL/min

    p1 is estimated.
    p2 is fixed.
    SI is estimated directly, and p3 is reconstructed as p2 * SI.
    """
    I_of_t = interp1d(
        t_ins,
        ins_uU,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    def rhs(t, y):
        G, X, GI, E = y
        I = float(I_of_t(t))
        Ra = Ka * E

        dG = -(p1 + X) * G + p1 * Gb + Ra
        dX = -p2_fixed * X + p2_fixed * SI * (I - Ib)
        dGI = -KGI * GI
        dE = -Ka * E + KGI * GI
        return [dG, dX, dGI, dE]

    y0 = [Gb, 0.0, GI0, 0.0]

    sol = solve_ivp(
        rhs,
        (float(np.min(t_eval)), float(np.max(t_eval))),
        y0,
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    G, X, GI, E = sol.y
    Ra = Ka * E
    return G, X, GI, E, Ra


def insulinogenic_index(bg0: float, bg30: float, iri0: float, iri30: float) -> float:
    denom = bg30 - bg0
    if np.isclose(denom, 0.0):
        return np.nan
    return (iri30 - iri0) / denom


def matsuda_index(glucose_vals: np.ndarray, insulin_vals: np.ndarray) -> float:
    glucose_vals = np.asarray(glucose_vals, dtype=float)
    insulin_vals = np.asarray(insulin_vals, dtype=float)
    fpg = glucose_vals[0]
    firi = insulin_vals[0]
    mean_pg = np.mean(glucose_vals)
    mean_iri = np.mean(insulin_vals)
    denom = (fpg * firi) * (mean_pg * mean_iri)
    if denom <= 0:
        return np.nan
    return 10000.0 / math.sqrt(denom)


def fit_subject_p1_si_model(t_points: np.ndarray, g_obs: np.ndarray, i_points: np.ndarray) -> Dict[str, float]:
    """
    Fit p1 and SI while fixing p2.

    Estimated parameters:
        p1, SI, KGI, Ka, GI0

    The model is fitted only to observed OGTT time points (0, 30, 60, 90, and 120 min).
    A weak prior on p1 is added to reduce implausible compensation between p1 and Ra.
    """
    t_points = np.asarray(t_points, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    i_points = np.asarray(i_points, dtype=float)

    if np.any(~np.isfinite(g_obs)) or np.any(~np.isfinite(i_points)):
        raise ValueError("G_obs または I_points に欠損/非数があります。")

    gb = float(g_obs[t_points == 0][0])
    ib = float(i_points[t_points == 0][0])

    # x = [p1, SI, KGI, Ka, GI0]
    x0 = np.array([0.016, 6.7e-4, 0.02, 0.013, 1.0], dtype=float)

    # Bounds are intentionally narrower than the previous prototype.
    # p1 remains estimable, but implausible values are avoided.
    lb = np.array([0.005, 1e-6, 1e-4, 1e-4, 1e-4], dtype=float)
    ub = np.array([0.040, 5e-2, 0.20, 0.20, 1e4], dtype=float)

    def resid(x):
        p1, si, kgi, ka, gi0 = x
        g_pred, *_ = simulate_p1_si_model(
            t_eval=t_points,
            t_ins=t_points,
            ins_uU=i_points,
            Gb=gb,
            Ib=ib,
            p1=p1,
            SI=si,
            KGI=kgi,
            Ka=ka,
            GI0=gi0,
            p2_fixed=P2_FIXED,
        )

        # Scale glucose residuals to roughly unit scale.
        r_glu = (g_pred - g_obs) / GLUCOSE_SD

        # Weak log-scale prior for p1 centered around 0.016 min^-1.
        # This is not a fixed value; it only discourages extreme compensation.
        r_p1_prior = np.array([0.20 * np.log(p1 / P1_PRIOR) / np.log(2.0)])

        return np.concatenate([r_glu, r_p1_prior])

    res = least_squares(
        resid,
        x0=x0,
        bounds=(lb, ub),
        max_nfev=5000,
        method="trf",
        x_scale="jac",
    )

    p1, si, kgi, ka, gi0 = res.x
    p2 = P2_FIXED
    p3 = p2 * si
    gi_half = np.log(2.0) / ka

    g_pred, x_pred, gi_pred, e_pred, ra_pred = simulate_p1_si_model(
        t_eval=t_points,
        t_ins=t_points,
        ins_uU=i_points,
        Gb=gb,
        Ib=ib,
        p1=p1,
        SI=si,
        KGI=kgi,
        Ka=ka,
        GI0=gi0,
        p2_fixed=P2_FIXED,
    )

    g_dense, x_dense, gi_dense, e_dense, ra_dense = simulate_p1_si_model(
        t_eval=DENSE_T,
        t_ins=t_points,
        ins_uU=i_points,
        Gb=gb,
        Ib=ib,
        p1=p1,
        SI=si,
        KGI=kgi,
        Ka=ka,
        GI0=gi0,
        p2_fixed=P2_FIXED,
    )

    return {
        "model_version": "p1_estimated_SI_estimated_p2_fixed_v2_5timepoint_no15",
        "p1_GE": p1,
        "p2": p2,
        "p2_fixed": p2,
        "p3": p3,
        "SI": si,
        "KGI": kgi,
        "Ka": ka,
        "GI0": gi0,
        "GI_half_life_min": gi_half,
        "cost": res.cost,
        "success": bool(res.success),
        "nfev": res.nfev,
        "Gb": gb,
        "Ib": ib,
        "G_pred": g_pred,
        "X_pred": x_pred,
        "GI_pred": gi_pred,
        "E_pred": e_pred,
        "Ra_pred": ra_pred,
        "G_dense": g_dense,
        "X_dense": x_dense,
        "GI_dense": gi_dense,
        "E_dense": e_dense,
        "Ra_dense": ra_dense,
        "message": res.message,
    }


def run_single_subject(subject_id: str, glucose_5pt: List[float], insulin_5pt: List[float]) -> FitSummary:
    glucose_5pt = np.asarray(glucose_5pt, dtype=float)
    insulin_5pt = np.asarray(insulin_5pt, dtype=float)

    g_fit = glucose_5pt.copy()
    i_fit = insulin_5pt.copy()

    fit = fit_subject_p1_si_model(T_FIT, g_fit, i_fit)

    igi = insulinogenic_index(glucose_5pt[0], glucose_5pt[1], insulin_5pt[0], insulin_5pt[1])
    matsuda = matsuda_index(glucose_5pt, insulin_5pt)
    odi = igi * matsuda if np.isfinite(igi) and np.isfinite(matsuda) else np.nan

    metrics = {
        "ID": subject_id,
        "model_version": fit["model_version"],
        "p1_GE": fit["p1_GE"],
        "p2": fit["p2"],
        "p2_fixed": fit["p2_fixed"],
        "p3": fit["p3"],
        "SI": fit["SI"],
        "KGI": fit["KGI"],
        "Ka": fit["Ka"],
        "GI0": fit["GI0"],
        "GI_half_life_min": fit["GI_half_life_min"],
        "Insulinogenic_index": igi,
        "Matsuda_index": matsuda,
        "oDI": odi,
        "cost": fit["cost"],
        "success": fit["success"],
        "nfev": fit["nfev"],
        "Gb": fit["Gb"],
        "Ib": fit["Ib"],
        "fit_message": fit["message"],
    }

    fit_table = pd.DataFrame({
        "Time_min": T_FIT,
        "G_obs": g_fit,
        "I_obs": i_fit,
        "G_pred": fit["G_pred"],
        "X_pred": fit["X_pred"],
        "GI_pred": fit["GI_pred"],
        "E_pred": fit["E_pred"],
        "Ra_pred": fit["Ra_pred"],
        "Is_observed_point": True,
    })

    dense_table = pd.DataFrame({
        "Time_min": DENSE_T,
        "G_pred": fit["G_dense"],
        "X_pred": fit["X_dense"],
        "GI_pred": fit["GI_dense"],
        "E_pred": fit["E_dense"],
        "Ra_pred": fit["Ra_dense"],
    })

    return FitSummary(metrics=metrics, fit_table=fit_table, dense_table=dense_table)


def validate_input_frame(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"入力ファイルに必要列がありません: {missing}")


def run_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, bytes]]:
    validate_input_frame(df)
    df = df.copy()
    df.columns = df.columns.str.strip()

    results = []
    fit_tables = {}
    dense_tables = {}

    for _, r in df.iterrows():
        subject_id = str(r["ID"])
        try:
            glucose_5pt = [r[c] for c in BG_COLS]
            insulin_5pt = [r[c] for c in IRI_COLS]
            summary = run_single_subject(subject_id, glucose_5pt, insulin_5pt)
            row = dict(summary.metrics)

            for _, rr in summary.fit_table.iterrows():
                t = int(rr["Time_min"])
                row[f"G_obs_{t}"] = rr["G_obs"]
                row[f"I_obs_{t}"] = rr["I_obs"]
                row[f"G_pred_{t}"] = rr["G_pred"]
                row[f"Ra_{t}"] = rr["Ra_pred"]

            results.append(row)
            fit_tables[subject_id] = summary.fit_table
            dense_tables[subject_id] = summary.dense_table
        except Exception as e:
            results.append({
                "ID": subject_id,
                "success": False,
                "error_message": str(e),
            })

    res_df = pd.DataFrame(results)

    main_cols = [
        "ID", "model_version", "p1_GE", "p2", "p2_fixed", "p3", "SI",
        "KGI", "Ka", "GI0", "GI_half_life_min", "Insulinogenic_index", "Matsuda_index",
        "oDI", "cost", "success", "nfev", "Gb", "Ib", "fit_message",
        "G_obs_0", "G_obs_30", "G_obs_60", "G_obs_90", "G_obs_120",
        "I_obs_0", "I_obs_30", "I_obs_60", "I_obs_90", "I_obs_120",
        "G_pred_0", "G_pred_30", "G_pred_60", "G_pred_90", "G_pred_120",
        "Ra_0", "Ra_30", "Ra_60", "Ra_90", "Ra_120", "error_message"
    ]
    existing_main = [c for c in main_cols if c in res_df.columns]
    others = [c for c in res_df.columns if c not in existing_main]
    res_df = res_df[existing_main + others]

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="FitResults", index=False)
        for subject_id, table in fit_tables.items():
            sheet = f"fit_{subject_id}"[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
        for subject_id, table in dense_tables.items():
            sheet = f"dense_{subject_id}"[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
    excel_bytes = excel_buffer.getvalue()

    csv_bytes = res_df.to_csv(index=False).encode("utf-8-sig")
    return res_df, {
        "batch_results.xlsx": excel_bytes,
        "batch_results.csv": csv_bytes,
    }
