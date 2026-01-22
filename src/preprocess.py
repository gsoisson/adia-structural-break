import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.config import (
    CLIP_QLOW,
    CLIP_QHIGH,
    CLIP_MIN_WIDTH,
    CLIP_DEFAULT_BAND,
    S0_FLOOR,
    S_DZ_FLOOR,
    S_ABS_FLOOR,
    S_DD_FLOOR,
    EPS,
    FEAT_CACHE_DIR,
)


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────


def pad_periods(X, period, max_len):
    """Return padded 2D array for one period (before/after)."""
    grouped = X.loc[X["period"] == period].groupby("id")["value"].apply(np.array)
    n_series = len(grouped)
    arr = np.full((n_series, max_len), np.nan, dtype=np.float32)
    for i, g in enumerate(grouped):
        arr[i, : len(g)] = g
    return arr, grouped.index


def mad(arr, axis=1):
    """Median absolute deviation along axis."""
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(arr - med), axis=axis, keepdims=True)


def winsorize_pair(
    before,
    after,
    qlow=CLIP_QLOW,
    qhigh=CLIP_QHIGH,
    min_width=CLIP_MIN_WIDTH,  # required (qh-ql) span in standardized units
    default_band=CLIP_DEFAULT_BAND,
):
    """
    Winsorize both segments using BEFORE per-ID cutoffs; fallback to default_band if BEFORE span collapses.
    Inputs should already be standardized (z).
    """
    # per-ID BEFORE quantiles
    ql_id = np.nanquantile(before, qlow, axis=1, keepdims=True)
    qh_id = np.nanquantile(before, qhigh, axis=1, keepdims=True)

    width = qh_id - ql_id
    wide_enough = np.isfinite(width) & (width >= min_width)

    # choose fallback band
    ql_default, qh_default = -float(default_band), float(default_band)

    ql = np.where(wide_enough, ql_id, ql_default)
    qh = np.where(wide_enough, qh_id, qh_default)

    wb = np.where(np.isnan(before), np.nan, np.clip(before, ql, qh))
    wa = np.where(np.isnan(after), np.nan, np.clip(after, ql, qh))
    return wb.astype(np.float32), wa.astype(np.float32)


def fast_detrend_ols(arr, mean_center=True):
    """
    Vectorized linear detrend per row with NaN masks.
    Returns mean-centered residuals (float32).
    """
    y = arr.astype(np.float32, copy=False)
    _, T = y.shape

    # x-axis: 0..T-1, then center per row to improve numerics
    t = np.arange(T, dtype=np.float32)[None, :]  # (1, T)
    mask = ~np.isnan(y)  # (n, T)
    cnt = mask.sum(axis=1, keepdims=True).astype(np.float32)

    t_sum = (mask * t).sum(axis=1, keepdims=True)  # Σ t_i
    y_sum = np.nansum(y, axis=1, keepdims=True)  # Σ y_i

    t_bar = t_sum / cnt  # \bar t
    y_bar = y_sum / cnt  # \bar y

    tc = t - t_bar  # center x
    yc = np.where(mask, y - y_bar, 0.0)  # center y where valid

    num = (tc * yc * mask).sum(axis=1, keepdims=True)  # Σ (tc * yc)
    den = (tc * tc * mask).sum(axis=1, keepdims=True)  # Σ (tc^2)
    den = np.where(den <= EPS, EPS, den)  # floor

    slope = num / den
    intercept = y_bar - slope * t_bar

    yhat = intercept + slope * t
    resid = np.where(mask, y - yhat, np.nan)

    if mean_center:
        resid = resid - np.nanmean(resid, axis=1, keepdims=True)

    return resid.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────


def process_series(X_train):
    """Heavy NumPy preprocessing for the entire dataset."""

    # --- Build padded arrays
    max_before = X_train.loc[X_train["period"] == 0].groupby("id").size().max()
    max_after = X_train.loc[X_train["period"] == 1].groupby("id").size().max()
    before_arr, ids = pad_periods(X_train, 0, max_before)
    after_arr, _ = pad_periods(X_train, 1, max_after)

    # --- Standardization by before stats (robust)
    m0 = np.nanmedian(before_arr, axis=1, keepdims=True)
    s0 = 1.4826 * mad(before_arr)
    s0 = np.maximum(s0, S0_FLOOR)
    z_before = (before_arr - m0) / s0
    z_after = (after_arr - m0) / s0

    # --- Winsorized standardized (replaces fixed np.clip)
    zc_before, zc_after = winsorize_pair(z_before, z_after)

    # --- Detrended (per segment) on winsorized standardized
    zd_before = fast_detrend_ols(zc_before)
    zd_after = fast_detrend_ols(zc_after)

    # --- Segment-aware diffs on standardized (pre-winsor)
    dz_before = np.diff(z_before, axis=1, prepend=z_before[:, [0]])
    dz_after = np.diff(z_after, axis=1, prepend=z_after[:, [0]])

    m_dz = np.nanmedian(dz_before, axis=1, keepdims=True)
    s_dz = 1.4826 * mad(dz_before)
    s_dz = np.maximum(s_dz, S_DZ_FLOOR)

    d_before = (dz_before - m_dz) / s_dz
    d_after = (dz_after - m_dz) / s_dz

    # --- Winsorized diffs (replaces fixed clip on diffs)
    dc_before, dc_after = winsorize_pair(d_before, d_after)

    # --- Detrend of winsorized diffs
    dm_before = fast_detrend_ols(dc_before)
    dm_after = fast_detrend_ols(dc_after)

    # --- Absolute diffs (from normalized diffs)
    a_before = np.abs(d_before)
    a_after = np.abs(d_after)

    # Winsorize absolute diffs by before’s absolute-diff quantiles
    ac_before, ac_after = winsorize_pair(a_before, a_after)

    # Detrend of winsorized absolute diffs
    am_before = fast_detrend_ols(ac_before)
    am_after = fast_detrend_ols(ac_after)

    # --- Absolute values of z (pre-winsor)
    absz_before = np.abs(z_before)
    absz_after = np.abs(z_after)

    m_abs = np.nanmedian(absz_before, axis=1, keepdims=True)
    s_abs = 1.4826 * mad(absz_before)
    s_abs = np.maximum(s_abs, S_ABS_FLOOR)
    abs_std_before = (absz_before - m_abs) / s_abs
    abs_std_after = (absz_after - m_abs) / s_abs

    # Winsorize normalized |z|
    abs_c_before, abs_c_after = winsorize_pair(abs_std_before, abs_std_after)

    # Detrend of winsorized |z| normalized
    abs_m_before = fast_detrend_ols(abs_c_before)
    abs_m_after = fast_detrend_ols(abs_c_after)

    # Build z^2, winsorize using BEFORE quantiles, then detrend per segment
    sq_before = np.square(z_before)
    sq_after = np.square(z_after)

    sqc_before, sqc_after = winsorize_pair(sq_before, sq_after)
    sqm_before = fast_detrend_ols(sqc_before)
    sqm_after = fast_detrend_ols(sqc_after)

    # --- Second differences (curvature) on standardized z, per segment (no cross-period leakage)
    dd_raw_before = np.diff(dz_before, axis=1, prepend=dz_before[:, [0]])
    dd_raw_after = np.diff(dz_after, axis=1, prepend=dz_after[:, [0]])

    # Robust standardization by BEFORE stats (median/MAD), reuse the diff floor to be conservative
    m_dd = np.nanmedian(dd_raw_before, axis=1, keepdims=True)
    s_dd = 1.4826 * mad(dd_raw_before)
    s_dd = np.maximum(s_dd, S_DD_FLOOR)

    dd_before = (dd_raw_before - m_dd) / s_dd
    dd_after = (dd_raw_after - m_dd) / s_dd

    # Winsorize second differences using BEFORE-based cutoffs
    ddc_before, ddc_after = winsorize_pair(dd_before, dd_after)

    # Detrend per segment (OLS) on winsorized second differences
    ddm_before = fast_detrend_ols(ddc_before)
    ddm_after = fast_detrend_ols(ddc_after)

    # --- Stitch back into MultiIndex DataFrame
    out_list = []
    for i, id_val in enumerate(ids):
        Lb = np.count_nonzero(~np.isnan(before_arr[i]))
        La = np.count_nonzero(~np.isnan(after_arr[i]))
        n_total = Lb + La
        time_index = np.arange(n_total)

        df = pd.DataFrame(
            {
                "original": np.r_[before_arr[i, :Lb], after_arr[i, :La]],
                "period": np.r_[np.zeros(Lb, dtype=int), np.ones(La, dtype=int)],
                "standardized": np.r_[z_before[i, :Lb], z_after[i, :La]],
                "clipped": np.r_[zc_before[i, :Lb], zc_after[i, :La]],
                "detrended": np.r_[zd_before[i, :Lb], zd_after[i, :La]],
                "diff_standardized": np.r_[d_before[i, :Lb], d_after[i, :La]],
                "diff_detrended": np.r_[dm_before[i, :Lb], dm_after[i, :La]],
                "absdiff_detrended": np.r_[am_before[i, :Lb], am_after[i, :La]],
                "absval_detrended": np.r_[abs_m_before[i, :Lb], abs_m_after[i, :La]],
                "squared_detrended": np.r_[sqm_before[i, :Lb], sqm_after[i, :La]],
                "diff2_standardized": np.r_[dd_before[i, :Lb], dd_after[i, :La]],
                "diff2_detrended": np.r_[ddm_before[i, :Lb], ddm_after[i, :La]],
            },
            index=pd.MultiIndex.from_product(
                [[id_val], time_index], names=["id", "time"]
            ),
        )
        out_list.append(df)

    return pd.concat(out_list, axis=0)


# ─────────────────────────────────────────────────────────────────────
# File-handling wrapper
# ─────────────────────────────────────────────────────────────────────
def _latest_cache(prefix: str):
    files = sorted(
        FEAT_CACHE_DIR.glob(f"{prefix}_*.parquet"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    latest = files[0] if files else None
    return latest


def _save_cache(df: pd.DataFrame, prefix: str) -> Path:
    ts = datetime.now().strftime("%m%d_%H%M")
    path = FEAT_CACHE_DIR / f"{prefix}_{ts}.parquet"
    df.to_parquet(path)
    return path


def detect_non_finite(feats: pd.DataFrame):
    arr = feats.to_numpy(dtype=np.float32, copy=False)
    mask = ~np.isfinite(arr)
    if mask.any():
        r, c = np.where(mask)
        for i in range(min(5, len(r))):
            print(
                f"  at row={feats.index[r[i]]}, col={feats.columns[c[i]]}, val={arr[r[i], c[i]]}"
            )
    return


def build_preprocessed(X_train, force=False, inference=False):
    """Check cache, run process_series if needed, and save."""
    prefix = "preprocessed"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    out = process_series(X_train)

    # Sanity check
    detect_non_finite(out)

    if not inference:
        _save_cache(out, prefix)
    return out
