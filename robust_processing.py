from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from pid_core import (
    FOPDTModel,
    PIDTuning,
    SimulationConfig,
    tuning_imc_simc,
    tuning_ziegler_nichols_openloop,
    tuning_cohen_coon,
    simulate_closed_loop,
)

OUTLIER_METHOD_LABELS = {
    "auto": "Auto",
    "none": "Sem exclusão",
    "hampel": "Hampel local",
    "robust_z": "Robust Z-score",
    "iqr_diff": "IQR nas diferenças",
    "rolling_iqr": "Rolling IQR",
    "global_mad": "MAD global",
}

MODEL_LABELS = {
    "auto": "Auto",
    "fopdt": "FOPDT",
    "sopdt": "SOPDT",
    "arx22": "ARX(2,2)",
    "arx33": "ARX(3,3)",
}

TUNING_METHOD_LABELS = {
    "compare_all": "Comparar todos",
    "imc_simc": "IMC/SIMC",
    "ziegler_nichols": "Ziegler-Nichols",
    "cohen_coon": "Cohen-Coon",
}


@dataclass
class MethodResult:
    method: str
    flagged_pct: float
    flagged_points: int
    validation_rmse: float
    validation_fit_pct: float
    validation_r2: float


@dataclass
class ModelResult:
    model_name: str
    validation_rmse: float
    validation_fit_pct: float
    validation_r2: float
    details: Dict[str, Any]


@dataclass
class EquivalentModel:
    source_model: str
    gain: float
    tau: float
    dead_time: float

    def to_fopdt(self) -> FOPDTModel:
        return FOPDTModel(gain=float(self.gain), tau=float(self.tau), dead_time=float(self.dead_time))


@dataclass
class AnalysisPackage:
    meta: Dict[str, Any]
    raw_data: pd.DataFrame
    clean_data: pd.DataFrame
    outlier_methods: List[MethodResult]
    selected_outlier_method: str
    model_results: List[ModelResult]
    selected_model_name: str
    equivalent_model: EquivalentModel
    performance_metrics: Dict[str, float]
    loop_score: float
    grade: str
    reasons: List[str]
    strengths: List[str]
    tuning_table: pd.DataFrame
    simulation_results: Dict[str, Dict[str, Any]]
    model_validation_series: Dict[str, pd.DataFrame]
    outlier_config: Dict[str, Any]


def _local_hampel(x: np.ndarray, window: int = 31, n_sigma: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    med = median_filter(x, size=max(3, int(window) | 1), mode="nearest")
    abs_dev = np.abs(x - med)
    mad = median_filter(abs_dev, size=max(3, int(window) | 1), mode="nearest")
    sigma = 1.4826 * np.maximum(mad, 1e-9)
    flag = abs_dev > n_sigma * sigma
    y = x.astype(float).copy()
    y[flag] = np.nan
    return y, flag


def _robust_z_filter(x: np.ndarray, window: int = 31, z: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
    trend = median_filter(x, size=max(3, int(window) | 1), mode="nearest")
    resid = x - trend
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    rz = 0.6745 * (resid - med) / max(mad, 1e-9)
    flag = np.abs(rz) > z
    y = x.astype(float).copy()
    y[flag] = np.nan
    return y, flag


def _diff_iqr_filter(x: np.ndarray, mult: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.diff(x, prepend=x[0])
    q1, q3 = np.nanquantile(dx, [0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    flag = (dx < lo) | (dx > hi)
    y = x.astype(float).copy()
    y[flag] = np.nan
    return y, flag


def _rolling_iqr_filter(x: np.ndarray, window: int = 41, mult: float = 2.8) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(x.astype(float))
    q1 = s.rolling(window=window, center=True, min_periods=max(5, window // 4)).quantile(0.25)
    q3 = s.rolling(window=window, center=True, min_periods=max(5, window // 4)).quantile(0.75)
    iqr = (q3 - q1).bfill().ffill()
    lo = q1 - mult * iqr
    hi = q3 + mult * iqr
    flag = ((s < lo) | (s > hi)).fillna(False).to_numpy()
    y = s.to_numpy().copy()
    y[flag] = np.nan
    return y, flag


def _global_mad_filter(x: np.ndarray, n_sigma: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * max(mad, 1e-9)
    flag = np.abs(x - med) > n_sigma * sigma
    y = x.astype(float).copy()
    y[flag] = np.nan
    return y, flag


def _finalize_cleaning(y: np.ndarray, strategy: str = "interpolate") -> np.ndarray:
    s = pd.Series(y.astype(float))
    if strategy == "drop":
        return s.to_numpy()
    return s.interpolate(limit_direction="both").to_numpy()


def _severity_window(base: int, aggressiveness: float) -> int:
    if aggressiveness <= 0:
        aggressiveness = 1.0
    scale = max(0.5, min(3.0, aggressiveness))
    w = int(round(base / scale))
    return max(5, w | 1)


def _apply_one_method(x: np.ndarray, method: str, aggressiveness: float) -> Tuple[np.ndarray, np.ndarray]:
    scale = max(0.5, min(3.0, aggressiveness))
    if method == "hampel":
        return _local_hampel(x, window=_severity_window(31, scale), n_sigma=max(1.8, 3.8 / scale))
    if method == "robust_z":
        return _robust_z_filter(x, window=_severity_window(31, scale), z=max(2.0, 4.8 / scale))
    if method == "iqr_diff":
        return _diff_iqr_filter(x, mult=max(1.2, 3.2 / scale))
    if method == "rolling_iqr":
        return _rolling_iqr_filter(x, window=_severity_window(41, scale), mult=max(1.2, 3.0 / scale))
    if method == "global_mad":
        return _global_mad_filter(x, n_sigma=max(2.0, 5.5 / scale))
    return x.astype(float).copy(), np.zeros(len(x), dtype=bool)


def _to_seconds(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return (pd.to_datetime(series) - pd.to_datetime(series).iloc[0]).dt.total_seconds()
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals - vals.iloc[0]
    return vals


def prepare_data(raw_df: pd.DataFrame, time_col: str, pv_col: str, sp_col: Optional[str], mv_col: Optional[str]) -> pd.DataFrame:
    selected = [time_col, pv_col] + ([sp_col] if sp_col else []) + ([mv_col] if mv_col else [])
    data = raw_df[selected].copy()
    rename = {time_col: "time", pv_col: "pv"}
    if sp_col:
        rename[sp_col] = "sp"
    if mv_col:
        rename[mv_col] = "mv"
    data = data.rename(columns=rename)
    data["time"] = pd.to_datetime(data["time"], errors="coerce", format="mixed")
    if data["time"].isna().all():
        data["time"] = _to_seconds(raw_df[time_col])
    for col in ["pv", "sp", "mv"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    keep = ["time", "pv"] + (["sp"] if "sp" in data.columns else []) + (["mv"] if "mv" in data.columns else [])
    data = data[keep].dropna().reset_index(drop=True)
    if np.issubdtype(data["time"].dtype, np.datetime64):
        data = data.sort_values("time").reset_index(drop=True)
    else:
        data["time"] = pd.to_numeric(data["time"], errors="coerce")
        data = data.sort_values("time").reset_index(drop=True)
    return data


def preprocess_for_method(
    data: pd.DataFrame,
    method: str,
    resample_rule: str = "60s",
    apply_to: str = "pv_mv",
    mv_limits: Tuple[float, float] = (0.0, 100.0),
    aggressiveness: float = 1.0,
    passes: int = 1,
    strategy: str = "interpolate",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    work = data.copy()
    time_dt = np.issubdtype(work["time"].dtype, np.datetime64)

    raw_sat_flag = np.zeros(len(work), dtype=bool)
    if "mv" in work.columns:
        raw_sat_flag = ((work["mv"] < mv_limits[0]) | (work["mv"] > mv_limits[1])).to_numpy()
        work["mv"] = work["mv"].clip(mv_limits[0], mv_limits[1])

    combined = raw_sat_flag.copy()
    for col in ["pv", "mv"]:
        if col not in work.columns:
            continue
        if apply_to == "pv" and col != "pv":
            continue
        if apply_to == "mv" and col != "mv":
            continue
        if apply_to == "none" or method == "none":
            continue
        x = work[col].to_numpy(dtype=float)
        col_flag = np.zeros(len(x), dtype=bool)
        for _ in range(max(1, int(passes))):
            y, flag = _apply_one_method(x, method, aggressiveness)
            col_flag |= flag
            x = _finalize_cleaning(y, strategy=strategy)
        work[col] = x
        combined |= col_flag

    work["outlier_flag"] = combined.astype(int)
    if strategy == "drop":
        filtered = work.loc[work["outlier_flag"] == 0].copy()
        if len(filtered) < max(20, len(work) * 0.2):
            filtered = work.copy()
    else:
        filtered = work.copy()

    if time_dt:
        ds = filtered.set_index("time").resample(resample_rule).mean(numeric_only=True).interpolate().reset_index()
        ds["time_s"] = (ds["time"] - ds["time"].iloc[0]).dt.total_seconds()
    else:
        step = float(str(resample_rule).replace("s", "")) if str(resample_rule).endswith("s") else 60.0
        t = pd.to_numeric(filtered["time"], errors="coerce").to_numpy(dtype=float)
        t = t - t[0]
        new_t = np.arange(0.0, t[-1] + step, step)
        out = {"time_s": new_t}
        for col in [c for c in filtered.columns if c not in {"time"}]:
            out[col] = np.interp(new_t, t, filtered[col].to_numpy(dtype=float))
        ds = pd.DataFrame(out)

    stats = {
        "flagged_pct": float(100.0 * work["outlier_flag"].mean()),
        "flagged_points": int(work["outlier_flag"].sum()),
        "raw_points": int(len(work)),
        "clean_points": int(len(ds)),
    }
    return work, ds, stats


def _delay_input(u: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return u.copy()
    return np.r_[np.full(d, u[0]), u[:-d]]


def _sim_fopdt_dev(u: np.ndarray, dt: float, K: float, tau: float, d: int) -> np.ndarray:
    ud = _delay_input(u, d)
    a = math.exp(-dt / max(tau, 1e-9))
    b = K * (1.0 - a)
    y = np.zeros_like(u, dtype=float)
    for k in range(1, len(u)):
        y[k] = a * y[k - 1] + b * ud[k - 1]
    return y


def _sim_sopdt_dev(u: np.ndarray, dt: float, K: float, t1: float, t2: float, d: int) -> np.ndarray:
    ud = _delay_input(u, d)
    a1 = math.exp(-dt / max(t1, 1e-9))
    b1 = 1.0 - a1
    a2 = math.exp(-dt / max(t2, 1e-9))
    b2 = 1.0 - a2
    x1 = np.zeros_like(u, dtype=float)
    x2 = np.zeros_like(u, dtype=float)
    for k in range(1, len(u)):
        x1[k] = a1 * x1[k - 1] + b1 * ud[k - 1]
        x2[k] = a2 * x2[k - 1] + b2 * x1[k]
    return K * x2


def fit_arx(train_df: pd.DataFrame, na: int = 2, nb: int = 2, dmax: int = 12) -> Dict[str, Any]:
    y = (train_df["pv"] - train_df["pv"].mean()).to_numpy()
    u = (train_df["mv"] - train_df["mv"].mean()).to_numpy()
    n = len(y)
    best = None
    for d in range(dmax + 1):
        start = max(na, d + nb)
        X = []
        Y = []
        for k in range(start, n):
            X.append([y[k - i] for i in range(1, na + 1)] + [u[k - d - j] for j in range(nb)])
            Y.append(y[k])
        X = np.array(X)
        Y = np.array(Y)
        theta = np.linalg.lstsq(X, Y, rcond=None)[0]
        yhat = np.zeros(n)
        yhat[:start] = y[:start]
        for k in range(start, n):
            row = [yhat[k - i] for i in range(1, na + 1)] + [u[k - d - j] for j in range(nb)]
            yhat[k] = float(np.dot(theta, row))
        rmse = float(np.sqrt(np.mean((y[start:] - yhat[start:]) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"na": na, "nb": nb, "d": d, "theta": theta, "rmse": rmse, "start": start}
    best["y_mean"] = float(train_df["pv"].mean())
    best["u_mean"] = float(train_df["mv"].mean())
    return best


def arx_predict_free_run(val_df: pd.DataFrame, model: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    y = (val_df["pv"] - model["y_mean"]).to_numpy()
    u = (val_df["mv"] - model["u_mean"]).to_numpy()
    na, nb, d = model["na"], model["nb"], model["d"]
    start = max(na, d + nb)
    yhat = np.zeros(len(y))
    yhat[:start] = y[:start]
    for k in range(start, len(y)):
        row = [yhat[k - i] for i in range(1, na + 1)] + [u[k - d - j] for j in range(nb)]
        yhat[k] = float(np.dot(model["theta"], row))
    return y, yhat, start


def fit_fopdt(train_df: pd.DataFrame) -> Dict[str, Any]:
    dt = float(np.median(np.diff(train_df["time_s"])))
    u = (train_df["mv"] - train_df["mv"].mean()).to_numpy()
    y = (train_df["pv"] - train_df["pv"].mean()).to_numpy()
    taus = np.geomspace(max(dt, 30.0), 3600.0, 24)
    best = None
    for d in range(0, 13):
        for tau in taus:
            basis = _sim_fopdt_dev(u, dt, 1.0, tau, d)
            K = float(np.dot(basis, y) / max(np.dot(basis, basis), 1e-12))
            yhat = K * basis
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            if best is None or rmse < best["rmse"]:
                best = {"K": K, "tau": float(tau), "d": int(d), "rmse": rmse}
    return best


def fit_sopdt(train_df: pd.DataFrame) -> Dict[str, Any]:
    dt = float(np.median(np.diff(train_df["time_s"])))
    u = (train_df["mv"] - train_df["mv"].mean()).to_numpy()
    y = (train_df["pv"] - train_df["pv"].mean()).to_numpy()
    t1s = np.geomspace(max(dt, 30.0), 900.0, 8)
    t2s = np.geomspace(max(2 * dt, 60.0), 3600.0, 9)
    best = None
    for d in range(0, 11):
        for t1 in t1s:
            for t2 in t2s:
                basis = _sim_sopdt_dev(u, dt, 1.0, t1, t2, d)
                K = float(np.dot(basis, y) / max(np.dot(basis, basis), 1e-12))
                yhat = K * basis
                rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
                if best is None or rmse < best["rmse"]:
                    best = {"K": K, "tau1": float(t1), "tau2": float(t2), "d": int(d), "rmse": rmse}
    return best


def _fit_metrics(y: np.ndarray, yhat: np.ndarray, start: int = 0) -> Tuple[float, float, float]:
    err = y[start:] - yhat[start:]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y[start:] - np.mean(y[start:])) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    fit_pct = 100.0 * (1.0 - np.linalg.norm(err) / max(np.linalg.norm(y[start:] - np.mean(y[start:])), 1e-12))
    return rmse, r2, float(fit_pct)


def compare_outlier_methods(
    raw_df: pd.DataFrame,
    resample_rule: str = "60s",
    apply_to: str = "pv_mv",
    mv_limits: Tuple[float, float] = (0.0, 100.0),
    aggressiveness: float = 1.0,
    passes: int = 1,
    strategy: str = "interpolate",
) -> Tuple[List[MethodResult], Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]]:
    methods = ["hampel", "robust_z", "iqr_diff", "rolling_iqr", "global_mad"]
    method_results: List[MethodResult] = []
    cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]] = {}
    for method in methods:
        raw_clean, ds, stats = preprocess_for_method(
            raw_df,
            method=method,
            resample_rule=resample_rule,
            apply_to=apply_to,
            mv_limits=mv_limits,
            aggressiveness=aggressiveness,
            passes=passes,
            strategy=strategy,
        )
        cache[method] = (raw_clean, ds, stats)
        cut = int(len(ds) * 0.7)
        train = ds.iloc[:cut].copy()
        val = ds.iloc[cut:].copy()
        if "mv" in ds.columns and len(val) > 10:
            arx = fit_arx(train, na=2, nb=2, dmax=min(12, max(2, len(train) // 20)))
            y, yhat, start = arx_predict_free_run(val, arx)
            rmse, r2, fit_pct = _fit_metrics(y, yhat, start)
        else:
            rmse, r2, fit_pct = np.nan, np.nan, np.nan
        method_results.append(
            MethodResult(
                method=method,
                flagged_pct=stats["flagged_pct"],
                flagged_points=stats["flagged_points"],
                validation_rmse=rmse,
                validation_fit_pct=fit_pct,
                validation_r2=r2,
            )
        )
    method_results.sort(key=lambda r: (np.nan_to_num(r.validation_rmse, nan=999.0), -np.nan_to_num(r.validation_fit_pct, nan=-999.0)))
    return method_results, cache


def compare_models(clean_df: pd.DataFrame) -> Tuple[List[ModelResult], Dict[str, pd.DataFrame]]:
    cut = int(len(clean_df) * 0.7)
    train = clean_df.iloc[:cut].copy()
    val = clean_df.iloc[cut:].copy()
    dt = float(np.median(np.diff(clean_df["time_s"])))
    u = (val["mv"] - train["mv"].mean()).to_numpy()
    y = (val["pv"] - train["pv"].mean()).to_numpy()
    time_val = val["time"].to_numpy() if "time" in val.columns else val["time_s"].to_numpy()

    results: List[ModelResult] = []
    validation: Dict[str, pd.DataFrame] = {}

    fop = fit_fopdt(train)
    yhat_f = _sim_fopdt_dev(u, dt, fop["K"], fop["tau"], fop["d"])
    rmse, r2, fit_pct = _fit_metrics(y, yhat_f, 0)
    results.append(ModelResult("FOPDT", rmse, fit_pct, r2, fop))
    validation["FOPDT"] = pd.DataFrame({"time": time_val, "pv_real": y + train["pv"].mean(), "pv_pred": yhat_f + train["pv"].mean()})

    sop = fit_sopdt(train)
    yhat_s = _sim_sopdt_dev(u, dt, sop["K"], sop["tau1"], sop["tau2"], sop["d"])
    rmse, r2, fit_pct = _fit_metrics(y, yhat_s, 0)
    results.append(ModelResult("SOPDT", rmse, fit_pct, r2, sop))
    validation["SOPDT"] = pd.DataFrame({"time": time_val, "pv_real": y + train["pv"].mean(), "pv_pred": yhat_s + train["pv"].mean()})

    a22 = fit_arx(train, na=2, nb=2, dmax=min(12, max(2, len(train) // 20)))
    yv, yh, start = arx_predict_free_run(val, a22)
    rmse, r2, fit_pct = _fit_metrics(yv, yh, start)
    results.append(ModelResult("ARX(2,2)", rmse, fit_pct, r2, a22))
    validation["ARX(2,2)"] = pd.DataFrame({"time": time_val[start:], "pv_real": yv[start:] + a22["y_mean"], "pv_pred": yh[start:] + a22["y_mean"]})

    a33 = fit_arx(train, na=3, nb=3, dmax=min(12, max(2, len(train) // 20)))
    yv, yh, start = arx_predict_free_run(val, a33)
    rmse, r2, fit_pct = _fit_metrics(yv, yh, start)
    results.append(ModelResult("ARX(3,3)", rmse, fit_pct, r2, a33))
    validation["ARX(3,3)"] = pd.DataFrame({"time": time_val[start:], "pv_real": yv[start:] + a33["y_mean"], "pv_pred": yh[start:] + a33["y_mean"]})

    results.sort(key=lambda r: (r.validation_rmse, -r.validation_fit_pct))
    return results, validation


def _arx_step_response(model: Dict[str, Any], horizon: int = 500) -> np.ndarray:
    na, nb, d = model["na"], model["nb"], model["d"]
    theta = model["theta"]
    y = np.zeros(horizon)
    u = np.ones(horizon)
    start = max(na, d + nb)
    for k in range(start, horizon):
        y[k] = sum(theta[i] * y[k - i - 1] for i in range(na)) + sum(theta[na + j] * u[k - d - j] for j in range(nb))
    return y


def _equivalent_from_step(step_y: np.ndarray, dt: float) -> EquivalentModel:
    gain = float(step_y[-1] - step_y[0])
    y0 = float(step_y[0])
    target2 = y0 + 0.02 * gain
    target63 = y0 + 0.632 * gain
    if gain >= 0:
        idx2 = int(np.argmax(step_y >= target2))
        idx63 = int(np.argmax(step_y >= target63))
    else:
        idx2 = int(np.argmax(step_y <= target2))
        idx63 = int(np.argmax(step_y <= target63))
    tau = max((idx63 * dt) - (idx2 * dt), dt)
    theta = max(idx2 * dt - 0.1 * tau, 0.0)
    return EquivalentModel(source_model="step_equivalent", gain=float(gain), tau=float(tau), dead_time=float(theta))


def choose_equivalent_model(selected_model_name: str, model_results: List[ModelResult], dt: float) -> EquivalentModel:
    chosen = next((m for m in model_results if m.model_name == selected_model_name), model_results[0])
    if chosen.model_name == "FOPDT":
        d = chosen.details
        return EquivalentModel(source_model="FOPDT", gain=float(d["K"]), tau=float(d["tau"]), dead_time=float(d["d"] * dt))
    if chosen.model_name == "SOPDT":
        d = chosen.details
        u = np.ones(500)
        step = _sim_sopdt_dev(u, dt, d["K"], d["tau1"], d["tau2"], d["d"])
        eq = _equivalent_from_step(step, dt)
        eq.source_model = "SOPDT equivalente"
        return eq
    if chosen.model_name.startswith("ARX"):
        step = _arx_step_response(chosen.details, horizon=500)
        eq = _equivalent_from_step(step, dt)
        eq.source_model = chosen.model_name + " equivalente"
        return eq
    return EquivalentModel(source_model="Fallback", gain=0.01, tau=300.0, dead_time=60.0)


def build_tuning_suite(eq_model: EquivalentModel, controller_type: str = "PI", tuning_selection: str = "compare_all") -> pd.DataFrame:
    model = eq_model.to_fopdt()
    controller_type = controller_type.upper()
    rows: List[Dict[str, Any]] = []

    if tuning_selection in {"compare_all", "imc_simc"}:
        tau = max(model.tau, 1e-9)
        theta = max(model.dead_time, 0.0)
        lam_rob = max(3.0 * theta, 0.8 * tau)
        lam_bal = max(theta, 0.35 * tau)
        imc_rob = tuning_imc_simc(model, lam_rob, controller_type=controller_type, name="IMC/SIMC robusto")
        imc_bal = tuning_imc_simc(model, lam_bal, controller_type=controller_type, name="IMC/SIMC recomendado")
        rows.extend([imc_rob.to_dict(), imc_bal.to_dict()])

    if tuning_selection in {"compare_all", "ziegler_nichols"}:
        rows.append(tuning_ziegler_nichols_openloop(model, controller_type=controller_type, name="Ziegler-Nichols").to_dict())

    if tuning_selection in {"compare_all", "cohen_coon"}:
        rows.append(tuning_cohen_coon(model, controller_type=controller_type, name="Cohen-Coon").to_dict())

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ti_seconds"] = df["ti"]
    df["ki_per_s"] = np.where(df["ti"] > 0, df["kc"] / df["ti"], np.nan)
    df["pb_percent_if_normalized"] = np.where(df["kc"] > 0, 100.0 / df["kc"], np.nan)
    return df


def simulate_tuning_suite(
    eq_model: EquivalentModel,
    tuning_df: pd.DataFrame,
    step_size: float = 0.02,
    horizon: float = 3600.0,
    dt: float = 1.0,
    baseline_pv: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    sims: Dict[str, Dict[str, Any]] = {}
    direct_action = bool(eq_model.gain >= 0)
    for _, row in tuning_df.iterrows():
        tuning = PIDTuning.from_time_constants(
            name=str(row["name"]),
            controller_type=str(row["controller_type"]),
            kc=float(row["kc"]),
            ti=float(row["ti"] if np.isfinite(row["ti"]) else math.inf),
            td=float(row["td"]),
            method=str(row["method"]),
        )
        result = simulate_closed_loop(
            eq_model.to_fopdt(),
            tuning,
            SimulationConfig(
                horizon=horizon,
                dt=dt,
                setpoint=step_size,
                direct_action=direct_action,
            ),
        )
        sims[tuning.name] = {
            "tuning": tuning,
            "simulation": result,
            "direct_action": direct_action,
            "error_formula": "e = SP - PV" if direct_action else "e = PV - SP",
            "baseline_pv": baseline_pv,
        }
    return sims


def controller_convention_text(process_gain: float) -> Dict[str, str]:
    if process_gain >= 0:
        return {
            "error_formula": "e = SP - PV",
            "note": "Para ganho de processo positivo, a convenção recomendada para manter Kc positivo é e = SP - PV.",
        }
    return {
        "error_formula": "e = PV - SP",
        "note": "Para ganho de processo negativo, a convenção recomendada para manter Kc positivo é e = PV - SP.",
    }


def compute_loop_performance(clean_df: pd.DataFrame, mv_limits: Tuple[float, float] = (0.0, 100.0)) -> Tuple[Dict[str, float], float, str, List[str], List[str]]:
    err = clean_df["sp"] - clean_df["pv"] if "sp" in clean_df.columns else clean_df["pv"] - clean_df["pv"].median()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pv_std = float(clean_df["pv"].std())
    mv_move_mean = float(np.mean(np.abs(np.diff(clean_df["mv"])))) if "mv" in clean_df.columns and len(clean_df) > 1 else 0.0
    within_005 = float(np.mean(np.abs(err) <= 0.05)) if len(err) else 0.0
    within_010 = float(np.mean(np.abs(err) <= 0.10)) if len(err) else 0.0
    sat_frac = float(np.mean((clean_df["mv"] <= mv_limits[0] + 0.5) | (clean_df["mv"] >= mv_limits[1] - 0.5))) if "mv" in clean_df.columns else np.nan
    score = 10.0
    reasons: List[str] = []
    strengths: List[str] = []

    if rmse > 0.15:
        score -= 2.2
        reasons.append("Erro global alto entre SP e PV.")
    elif rmse > 0.10:
        score -= 1.5
        reasons.append("Erro moderado entre SP e PV.")
    else:
        strengths.append("Erro global relativamente baixo.")

    if pv_std > 0.14:
        score -= 1.7
        reasons.append("Variabilidade da PV elevada em relação ao patamar da malha.")
    elif pv_std > 0.09:
        score -= 0.9
        reasons.append("Variabilidade moderada da PV.")

    if mv_move_mean > 1.0:
        score -= 1.1
        reasons.append("Esforço elevado da MV para sustentar a qualidade atual.")
    elif mv_move_mean > 0.6:
        score -= 0.6
        reasons.append("Movimentação moderadamente alta da MV.")

    if not np.isnan(sat_frac):
        if sat_frac > 0.03:
            score -= 0.6
            reasons.append("A saída encostou com frequência nos limites operacionais.")
        elif sat_frac < 0.005:
            strengths.append("Sem evidência relevante de saturação.")

    if within_010 < 0.70:
        score -= 1.4
        reasons.append("Baixa permanência da PV dentro de ±0,10 do SP.")
    elif within_010 > 0.85:
        strengths.append("Boa permanência da PV na banda de ±0,10 do SP.")

    if within_005 < 0.40:
        score -= 0.5
        reasons.append("Baixa permanência da PV dentro de ±0,05 do SP.")

    score = float(np.clip(score, 0.0, 10.0))
    grade = "Excelente" if score >= 9 else "Muito boa" if score >= 8 else "Boa" if score >= 7 else "Atenção" if score >= 5.5 else "Crítica" if score >= 4 else "Muito crítica"
    metrics = {
        "rmse": rmse,
        "pv_std": pv_std,
        "mv_move_mean": mv_move_mean,
        "within_005_pct": 100.0 * within_005,
        "within_010_pct": 100.0 * within_010,
        "sat_frac_pct": 100.0 * sat_frac if not np.isnan(sat_frac) else np.nan,
        "mean_abs_error": float(np.mean(np.abs(err))),
        "n_points": float(len(clean_df)),
    }
    return metrics, score, grade, reasons, strengths


def analyze_loop(
    raw_df: pd.DataFrame,
    meta: Dict[str, Any],
    time_col: str,
    pv_col: str,
    sp_col: Optional[str],
    mv_col: Optional[str],
    outlier_method: str = "auto",
    resample_rule: str = "60s",
    apply_to: str = "pv_mv",
    model_family: str = "auto",
    tuning_selection: str = "compare_all",
    controller_type: str = "PI",
    step_size: float = 0.02,
    mv_limits: Tuple[float, float] = (0.0, 100.0),
    aggressiveness: float = 1.0,
    passes: int = 1,
    outlier_strategy: str = "interpolate",
) -> AnalysisPackage:
    data = prepare_data(raw_df, time_col, pv_col, sp_col, mv_col)
    methods, cache = compare_outlier_methods(
        data,
        resample_rule=resample_rule,
        apply_to=apply_to,
        mv_limits=mv_limits,
        aggressiveness=aggressiveness,
        passes=passes,
        strategy=outlier_strategy,
    )
    chosen_method = methods[0].method if outlier_method == "auto" else outlier_method
    if chosen_method == "none":
        raw_clean, clean_df, _ = preprocess_for_method(
            data,
            method="none",
            resample_rule=resample_rule,
            apply_to="none",
            mv_limits=mv_limits,
            aggressiveness=aggressiveness,
            passes=passes,
            strategy=outlier_strategy,
        )
    else:
        raw_clean, clean_df, _ = cache[chosen_method]
    if "mv" not in clean_df.columns:
        raise ValueError("Para esta versão robusta, a coluna MV medida é necessária para modelagem e comparação dos métodos de sintonia.")
    models, validation_series = compare_models(clean_df)
    selected_model = models[0].model_name if model_family == "auto" else {"fopdt": "FOPDT", "sopdt": "SOPDT", "arx22": "ARX(2,2)", "arx33": "ARX(3,3)"}.get(model_family, model_family)
    eq = choose_equivalent_model(selected_model, models, float(np.median(np.diff(clean_df["time_s"]))))
    perf_metrics, score, grade, reasons, strengths = compute_loop_performance(clean_df, mv_limits=mv_limits)
    tuning_df = build_tuning_suite(eq, controller_type=controller_type, tuning_selection=tuning_selection)
    baseline_pv = float(clean_df["pv"].median()) if len(clean_df) else 0.0
    sims = simulate_tuning_suite(eq, tuning_df, step_size=step_size, horizon=3600.0, baseline_pv=baseline_pv)
    return AnalysisPackage(
        meta=meta,
        raw_data=data,
        clean_data=clean_df,
        outlier_methods=methods,
        selected_outlier_method=chosen_method,
        model_results=models,
        selected_model_name=selected_model,
        equivalent_model=eq,
        performance_metrics=perf_metrics,
        loop_score=score,
        grade=grade,
        reasons=reasons,
        strengths=strengths,
        tuning_table=tuning_df,
        simulation_results=sims,
        model_validation_series=validation_series,
        outlier_config={
            "aggressiveness": aggressiveness,
            "passes": passes,
            "strategy": outlier_strategy,
            "apply_to": apply_to,
        },
    )
