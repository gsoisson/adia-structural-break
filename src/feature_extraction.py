from pathlib import Path
import math
import numpy as np
import pandas as pd
import pywt

from scipy.special import (
    gammaincc,
)  # for chi-square survival function (ARCH-LM p-values)


from src.preprocess import (
    _latest_cache,
    _save_cache,
    build_preprocessed,
    detect_non_finite,
)
from src.config import (
    FEAT_CACHE_DIR,
    EPS,
    TOP_K,
    QUANTILE_FINE_GRID,
    QUANTILE_COARSE_GRID,
    W_FANO,
    CROSSING_RATE_DEADBAND,
    ACF_MAX_LAG,
    LBQ_M,
    JS_QUANTILE_BINS,
    MMD_MAX_N,
    FREQ_BANDS,
    DWT_WAVELET,
    DWT_LEVEL,
    ENTROPY_M1,
    ENTROPY_M2,
    ENTROPY_TAU,
    BOUND_EDGE,
    BOUND_WINDOW_SIZES,
    BOUND_SKIP_AFTER,
    BOUND_ACF_MAX_LAG,
    ARCH_L,
    BOUND_OFFSETS,
    ROLL_WINDOWS,
    ROLL_MIN_POS_PER_HALF,
    ROLL_TOPK,
    EWVAR_HALFLIVES,
    AR_ORDER,
    AR_RIDGE_LAMBDA,
    AR_SCORE_CAP,
)

# ─────────────────────────────────────────────────────────────────────
# MOMENTS BLOCK
# ─────────────────────────────────────────────────────────────────────


def _group_series(
    df: pd.DataFrame, col: str, period: int
) -> pd.core.groupby.SeriesGroupBy:
    return df.loc[df["period"] == period, col].groupby(level="id")


def _log(x):
    return np.log(np.maximum(x, EPS))


def _std(s):
    return np.sqrt(np.nanmean((s - np.nanmean(s)) ** 2))


def _skew_kurt(s):
    # classic (population) skew and excess kurtosis; NaN-safe
    m = np.nanmean(s)
    v = np.nanmean((s - m) ** 2)
    std = np.sqrt(v)
    z = (s - m) / std
    skew = np.nanmean(z**3)
    kurt = np.nanmean(z**4) - 3.0
    return float(skew), float(kurt)


def _mad(s):
    med = np.nanmedian(s)
    return np.nanmedian(np.abs(s - med))


def _ols_slope(y):
    # slope of y vs t (0..n-1), NaN-safe
    n = len(y)
    t = np.arange(n, dtype=np.float32)
    t_bar = t.mean()
    y_bar = y.mean()
    num = np.sum((t - t_bar) * (y - y_bar))
    den = np.sum((t - t_bar) ** 2)
    if den <= EPS:
        return 0.0
    return float(num / den)


def _topk_mean(arr, k=TOP_K):
    if arr.size == 0:
        return np.nan
    k = min(k, arr.size)
    # np.partition is O(n); take largest k, then mean
    part = np.partition(arr, -k)[-k:]
    return float(np.nanmean(part))


def _bottomk_mean(arr, k=TOP_K):
    k = min(k, arr.size)
    part = np.partition(arr, k - 1)[:k]
    return float(np.nanmean(part))


def compute_moments_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    quantile_coarse_grid: list[float] = QUANTILE_COARSE_GRID,
) -> pd.DataFrame:
    # Load cache
    prefix = "moments"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # Group series
    x_col = "original"
    z_col = "standardized"
    clip_col = "clipped"

    x_b = _group_series(X_prep, x_col, 0)
    z_b = _group_series(X_prep, z_col, 0)
    z_a = _group_series(X_prep, z_col, 1)
    zc_b = _group_series(X_prep, clip_col, 0)
    zc_a = _group_series(X_prep, clip_col, 1)
    ids = z_b.size().index

    # FEATURE BUILDING BLOCKS

    # Base moments
    mean_b = z_b.mean()
    mean_a = z_a.mean()
    std_b = z_b.apply(_std)
    std_a = z_a.apply(_std)
    skew_b = z_b.apply(lambda s: _skew_kurt(s.values)[0])
    skew_a = z_a.apply(lambda s: _skew_kurt(s.values)[0])
    kurt_b = z_b.apply(lambda s: _skew_kurt(s.values)[1])
    kurt_a = z_a.apply(lambda s: _skew_kurt(s.values)[1])

    # Medians & MADs
    mad_orig_b = x_b.apply(lambda s: _mad(s.values))
    scaled_med_orig_b = x_b.median() / np.maximum(mad_orig_b, EPS)
    med_b = z_b.median()
    med_a = z_a.median()
    mad_b = z_b.apply(lambda s: _mad(s.values))
    mad_a = z_a.apply(lambda s: _mad(s.values))

    # Base quantiles
    qs = quantile_coarse_grid
    Q10_b, Q25_b, Q50_b, Q75_b, Q90_b = [z_b.quantile(q) for q in qs]
    Q10_a, Q25_a, Q50_a, Q75_a, Q90_a = [z_a.quantile(q) for q in qs]

    # Robust skew via central asymmetry
    rob_skew_b = _log((Q75_b - Q50_b) / (Q50_b - Q25_b + EPS))
    rob_skew_a = _log((Q75_a - Q50_a) / (Q50_a - Q25_a + EPS))

    # Robust kurtosis pieces: IQR, IDR
    iqr_b = Q75_b - Q25_b
    iqr_a = Q75_a - Q25_a
    idr_b = Q90_b - Q10_b
    idr_a = Q90_a - Q10_a

    # Trend (slope) on clipped/winsorized standardized series
    slope_b = zc_b.apply(lambda s: _ols_slope(s.values))
    slope_a = zc_a.apply(lambda s: _ols_slope(s.values))

    # FEATURE COMPUTATION

    # Robust location shift, Δmedian/MAD (on original) = median_after (on standardized)
    med_delta = med_a - med_b  # med_b should be 0

    # Robust scale shift
    mad_orig_b = _log(mad_orig_b)
    mad_logratio = _log((mad_a + EPS) / (mad_b + EPS))

    # Classic-vs-robust contrasts (per segment)
    mean_vs_med = mean_a - med_a - (mean_b - med_b)
    std_vs_mad = _log(std_a / (1.4826 * mad_a + EPS)) - _log(
        std_b / (1.4826 * mad_b + EPS)
    )
    skew_contrast = skew_a - rob_skew_a - skew_b - rob_skew_b
    kurt_contrast = (
        kurt_a - _log(idr_a / (iqr_a + EPS)) - (kurt_b - _log(idr_b / (iqr_b + EPS)))
    )

    # Slope
    slope_delta = slope_a - slope_b

    # Assemble
    df = pd.DataFrame(
        {
            # robust moments
            "scaled_med_orig_b": scaled_med_orig_b,
            "med_delta": med_delta,
            "mad_orig_b": mad_orig_b,
            "mad_logratio": mad_logratio,
            # contrasts
            "mean_vs_med": mean_vs_med,
            "std_vs_mad": std_vs_mad,
            "skew_contrast": skew_contrast,
            "kurt_contrast": kurt_contrast,
            # slope
            "slope_delta": slope_delta,
        },
        index=ids,
        dtype=np.float32,
    )
    if not inference:
        _save_cache(df, prefix)
    return df


# ─────────────────────────────────────────────────────────────────────
# QUANTILES BLOCK
# ─────────────────────────────────────────────────────────────────────


def compute_quantiles_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    quantile_fine_grid: list[float] = QUANTILE_FINE_GRID,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    # Load cache
    prefix = "quantiles"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # Group series
    z_col = "standardized"
    z_b = _group_series(X_prep, z_col, 0)
    z_a = _group_series(X_prep, z_col, 1)
    ids = z_b.size().index

    # FEATURE BUILDING BLOCKS

    # Base quantiles
    qs = quantile_fine_grid
    qb = z_b.quantile(q=qs).unstack(level=-1)
    qa = z_a.quantile(q=qs).unstack(level=-1)
    Q5_b, Q10_b, Q25_b, Q40_b, Q50_b, Q60_b, Q75_b, Q90_b, Q95_b = [qb[q] for q in qs]
    Q5_a, Q10_a, Q25_a, Q40_a, Q50_a, Q60_a, Q75_a, Q90_a, Q95_a = [qa[q] for q in qs]

    IQR_b = Q75_b - Q25_b
    IQR_a = Q75_a - Q25_a
    IDR_b = Q90_b - Q10_b
    IDR_a = Q90_a - Q10_a

    IDR_logratio = _log((IDR_a + EPS) / (IDR_b + EPS))

    # Tail asymmetry (robust skew)
    central_asym_b = _log((Q75_b - Q50_b) / (Q50_b - Q25_b + EPS))
    central_asym_a = _log((Q75_a - Q50_a) / (Q50_a - Q25_a + EPS))

    shoulder_asym_b = _log((Q90_b - Q75_b) / (Q25_b - Q10_b + EPS))
    shoulder_asym_a = _log((Q90_a - Q75_a) / (Q25_a - Q10_a + EPS))

    tail_asym_b = _log((Q95_b - Q90_b) / (Q10_b - Q5_b + EPS))
    tail_asym_a = _log((Q95_a - Q90_a) / (Q10_a - Q5_a + EPS))

    # Tail thickness / peakedness (robust kurtosis)
    central_weight_b = _log(IQR_b / (Q60_b - Q40_b + EPS))
    central_weight_a = _log(IQR_a / (Q60_a - Q40_a + EPS))

    shoulder_weight_b = _log(IDR_b / (IQR_b + EPS))
    shoulder_weight_a = _log(IDR_a / (IQR_a + EPS))

    tail_weight_b = _log((Q95_b - Q5_b) / (IDR_b + EPS))
    tail_weight_a = _log((Q95_a - Q5_a) / (IDR_a + EPS))

    # Moors-like variant: ((Q90 - Q60) - (Q40 - Q10)) / (Q75 - Q25)
    moors_b = ((Q90_b - Q60_b) - (Q40_b - Q10_b)) / (IQR_b + EPS)
    moors_a = ((Q90_a - Q60_a) - (Q40_a - Q10_a)) / (IQR_a + EPS)

    # Tail decay rates
    central_decay_b = _log((Q75_b - Q60_b) / (Q60_b - Q50_b + EPS))
    central_decay_a = _log((Q75_a - Q60_a) / (Q60_a - Q50_a + EPS))

    shoulder_decay_b = _log((Q90_b - Q75_b) / (Q75_b - Q60_b + EPS))
    shoulder_decay_a = _log((Q90_a - Q75_a) / (Q75_a - Q60_a + EPS))

    tail_decay_b = _log((Q95_b - Q90_b) / (Q90_b - Q75_b + EPS))
    tail_decay_a = _log((Q95_a - Q90_a) / (Q90_a - Q75_a + EPS))

    # Tail extremes (top/bottom k means) relative to IDR
    topk_b = z_b.apply(lambda s: _topk_mean(s.values, top_k))
    topk_a = z_a.apply(lambda s: _topk_mean(s.values, top_k))
    botk_b = z_b.apply(lambda s: _bottomk_mean(s.values, top_k))
    botk_a = z_a.apply(lambda s: _bottomk_mean(s.values, top_k))

    upper_ext_b = _log(np.maximum(topk_b, EPS) / (IDR_b + EPS))
    upper_ext_a = _log(np.maximum(topk_a, EPS) / (IDR_a + EPS))

    # For lower extension we expect botk to be negative; use -mean_bottom_k
    lower_ext_b = _log(np.maximum(-botk_b, EPS) / (IDR_b + EPS))
    lower_ext_a = _log(np.maximum(-botk_a, EPS) / (IDR_a + EPS))

    # Extreme tail asymmetry: log(mean_top_k / (-mean_bottom_k))
    # Guard denominator sign; if mean_bottom_k >= 0, push to EPS to avoid invalid log.
    eta_b = _log(np.maximum(topk_b, EPS) / np.maximum(-botk_b, EPS))
    eta_a = _log(np.maximum(topk_a, EPS) / np.maximum(-botk_a, EPS))

    # FEATURE COMPUTATION

    # Base quantiles
    Q5_delta = Q5_a - Q5_b
    Q10_delta = Q10_a - Q10_b
    Q25_delta = Q25_a - Q25_b
    Q40_delta = Q40_a - Q40_b
    Q60_delta = Q60_a - Q60_b
    Q75_delta = Q75_a - Q75_b
    Q90_delta = Q90_a - Q90_b
    Q95_delta = Q95_a - Q95_b

    # Tail asymmetry (robust skew)
    central_asym = central_asym_a - central_asym_b
    shoulder_asym = shoulder_asym_a - shoulder_asym_b
    tail_asym = tail_asym_a - tail_asym_b

    # Tail thickness / peakedness (robust kurtosis)
    central_weight = central_weight_a - central_weight_b
    shoulder_weight = shoulder_weight_a - shoulder_weight_b
    tail_weight = tail_weight_a - tail_weight_b
    moors = moors_a - moors_b

    # Tail decay rates
    central_decay = central_decay_a - central_decay_b
    shoulder_decay = shoulder_decay_a - shoulder_decay_b
    tail_decay = tail_decay_a - tail_decay_b

    # Tail extremes
    upper_ext = upper_ext_a - upper_ext_b
    lower_ext = lower_ext_a - lower_ext_b
    extreme_tail_asym = eta_a - eta_b

    # Assemble
    q_df = pd.DataFrame(
        {
            # base quantiles
            "Q5_delta": Q5_delta,
            "Q10_delta": Q10_delta,
            "Q25_delta": Q25_delta,
            "Q40_delta": Q40_delta,
            "Q60_delta": Q60_delta,
            "Q75_delta": Q75_delta,
            "Q90_delta": Q90_delta,
            "Q95_delta": Q95_delta,
            "IDR_logratio": IDR_logratio,
            # tail asym
            "central_asym": central_asym,
            "shoulder_asym": shoulder_asym,
            "tail_asym": tail_asym,
            # tail thickness
            "central_weight": central_weight,
            "shoulder_weight": shoulder_weight,
            "tail_weight": tail_weight,
            "moors": moors,
            # decay rates
            "central_decay": central_decay,
            "shoulder_decay": shoulder_decay,
            "tail_decay": tail_decay,
            # extremes
            "upper_ext": upper_ext,
            "lower_ext": lower_ext,
            "extreme_tail_asym": extreme_tail_asym,
        },
        index=ids,
        dtype=np.float32,
    )
    if not inference:
        _save_cache(q_df, prefix)
    return q_df


# ─────────────────────────────────────────────────────────────────────
# RATES BLOCK
# ─────────────────────────────────────────────────────────────────────


def jeffreys_logit(k: int, m: int) -> float:
    """Logit of Jeffreys-smoothed rate: p~ = (k+0.5)/(m+1)."""
    if m <= 0:
        raise ValueError
    p = (k + 0.5) / (m + 1.0)
    p = np.clip(p, EPS, 1 - EPS)
    return _log(p / (1.0 - p))


def exceedance_logits(
    z: np.ndarray, k_abs: float, k_fixed: float
) -> tuple[float, float]:
    """P(|z|>k_abs) and P(|z|>k_fixed) → Jeffreys logits."""
    m = z.size
    k1 = int(np.sum(np.abs(z) > k_abs))
    k2 = int(np.sum(np.abs(z) > k_fixed))
    return jeffreys_logit(k1, m), jeffreys_logit(k2, m)


def upper_lower_logits(
    z: np.ndarray, q_low: float, q_high: float
) -> tuple[float, float]:
    """P(z > q_high), P(z < q_low) → Jeffreys logits."""
    m = z.size
    ku = int(np.sum(z > q_high))
    kl = int(np.sum(z < q_low))
    return jeffreys_logit(ku, m), jeffreys_logit(kl, m)


def fano_burstiness(z: np.ndarray, thr_abs: float, w: int) -> float:
    """Fano factor of windowed counts of |z|>thr_abs over fixed window size w."""
    n = z.size
    n_w = (n // w) * w
    if n_w == 0:
        return 0.0
    blocks = z[:n_w].reshape(-1, w)
    hits = np.sum(np.abs(blocks) > thr_abs, axis=1).astype(np.float32)
    mu = hits.mean()
    if mu <= 0:
        return 0.0
    var = hits.var(ddof=0)
    return float(var / (mu + EPS))


def crossing_rates_logits(
    z: np.ndarray, Q: float, eps: float
) -> tuple[float, float, float]:
    """
    Up/Down/Total crossing rates (per transition), with deadband:
        up:   z_t <= Q - eps and z_{t+1} >= Q + eps
        down: z_t >= Q + eps and z_{t+1} <= Q - eps
    Returns Jeffreys-smoothed logits for up, down, total.
    """
    a = z[:-1]
    b = z[1:]
    m = a.size
    up = np.sum((a <= (Q - eps)) & (b >= (Q + eps)))
    dn = np.sum((a >= (Q + eps)) & (b <= (Q - eps)))
    tot = int(up + dn)
    return (
        jeffreys_logit(int(up), m),
        jeffreys_logit(int(dn), m),
        jeffreys_logit(tot, m),
    )


def median_crossing_asym(z: np.ndarray, Q: float, eps: float) -> float:
    """(UP_CR50 - DOWN_CR50) / (UP_CR50 + DOWN_CR50 + 1), using raw (unsmoothed) rates."""
    a = z[:-1]
    b = z[1:]
    up = np.sum((a <= (Q - eps)) & (b >= (Q + eps)))
    dn = np.sum((a >= (Q + eps)) & (b <= (Q - eps)))
    denom = up + dn
    if denom == 0:
        return 0.0
    return float((up - dn) / (denom + 1))  # +1 prevents +1 or -1 for actanh


def mean_log_res_time(z: np.ndarray, Q: float, eps: float) -> float:
    """
    Average time between *median* crossings (using deadband).
    Compute distances between consecutive (up or down) events; log of mean.
    """
    a, b = z[:-1], z[1:]
    m = a.size
    cross_idx = np.where(
        ((a <= Q - eps) & (b >= Q + eps)) | ((a >= Q + eps) & (b <= Q - eps))
    )[0]
    tot = cross_idx.size
    if tot == 0:
        return 0.0
    if tot == 1:
        # approx mean gap ≈ (m)/(tot+1)
        return _log(m / (tot + 1.0) + EPS)
    gaps = np.diff(cross_idx)
    return _log(np.mean(gaps) + EPS)


def fisher_delta(a: float, b: float, eps: float = EPS) -> float:
    """atanh(a) - atanh(b) with safe clamping for inputs in [-1,1]."""
    lo, hi = -1 + eps, 1 - eps
    return float(np.arctanh(np.clip(a, lo, hi)) - np.arctanh(np.clip(b, lo, hi)))


def compute_rates_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    quantile_fine_grid: list[float] = QUANTILE_FINE_GRID,
    w_fano: int = W_FANO,
    crossing_rate_deadband: float = CROSSING_RATE_DEADBAND,
) -> pd.DataFrame:
    """
    Rates block (after − before or log-ratios), computed per id via a single groupby-apply.
    Uses BEFORE quantiles as thresholds for both segments.
    """
    prefix = "rates"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # keep only what we need
    cols = ["standardized", "original", "period"]
    df = X_prep[cols].copy()

    def _one_id(g: pd.DataFrame) -> pd.Series:
        # split
        gb = g[g["period"] == 0]
        ga = g[g["period"] == 1]

        zb = gb["standardized"].to_numpy(np.float32, copy=False)
        za = ga["standardized"].to_numpy(np.float32, copy=False)
        xb = gb["original"].to_numpy(np.float32, copy=False)
        xa = ga["original"].to_numpy(np.float32, copy=False)

        # BEFORE thresholds
        qs = quantile_fine_grid
        Q5_b, _, Q25_b, _, Q50_b, _, Q75_b, _, Q95_b = np.quantile(zb, qs)

        # Extreme events (Jeffreys-smoothed logits)
        b1, b2 = exceedance_logits(zb, k_abs=Q95_b, k_fixed=3.0)
        a1, a2 = exceedance_logits(za, k_abs=Q95_b, k_fixed=3.0)

        u_b, l_b = upper_lower_logits(zb, q_low=Q5_b, q_high=Q95_b)
        u_a, l_a = upper_lower_logits(za, q_low=Q5_b, q_high=Q95_b)

        # Burstiness: Fano on |z|>Q95_before, then log-ratio
        fb = fano_burstiness(zb, thr_abs=Q95_b, w=w_fano)
        fa = fano_burstiness(za, thr_abs=Q95_b, w=w_fano)
        fano_logratio = _log((fa + EPS) / (fb + EPS))

        # 2) Crossing rates (deadband ε), keep totals at Q25/Q50/Q75
        _, _, cr25_b_t = crossing_rates_logits(zb, Q25_b, crossing_rate_deadband)
        _, _, cr25_a_t = crossing_rates_logits(za, Q25_b, crossing_rate_deadband)

        _, _, cr50_b_t = crossing_rates_logits(zb, Q50_b, crossing_rate_deadband)
        _, _, cr50_a_t = crossing_rates_logits(za, Q50_b, crossing_rate_deadband)

        _, _, cr75_b_t = crossing_rates_logits(zb, Q75_b, crossing_rate_deadband)
        _, _, cr75_a_t = crossing_rates_logits(za, Q75_b, crossing_rate_deadband)

        # Median crossing asymmetry (Fisher-z delta) & log mean residence time
        med_asym_b = median_crossing_asym(zb, Q50_b, crossing_rate_deadband)
        med_asym_a = median_crossing_asym(za, Q50_b, crossing_rate_deadband)
        med_cross_asym = fisher_delta(med_asym_a, med_asym_b)

        # CR decay
        cr_decay = (cr75_a_t - cr25_a_t) - (cr75_b_t - cr25_b_t)

        # 3) % zeros (original) → Jeffreys-logit delta
        kb, mb = int(np.isclose(xb, 0.0).sum()), xb.size
        ka, ma = int(np.isclose(xa, 0.0).sum()), xa.size
        pct_zeros = jeffreys_logit(ka, ma) - jeffreys_logit(kb, mb)

        return pd.Series(
            {
                "abs_excd_q95_logit_delta": a1 - b1,
                "abs_excd_3_logit_delta": a2 - b2,
                "up_excd_q95_logit_delta": u_a - u_b,
                "low_excd_q5_logit_delta": l_a - l_b,
                "fano_logratio": fano_logratio,
                "signflips_logit_delta": cr50_a_t - cr50_b_t,
                "CR25_logit_delta": cr25_a_t - cr25_b_t,
                "CR75_logit_delta": cr75_a_t - cr75_b_t,
                "med_cross_asym_fisher_delta": med_cross_asym,
                "cr_decay_logit_delta": cr_decay,
                "pct_zeros_logit_delta": pct_zeros,
            },
            dtype=np.float32,
        )

    out = (
        df.groupby(level="id", sort=False, group_keys=False)
        .apply(_one_id)
        .astype(np.float32)
    )

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# AUTOCORRELATION BLOCK
# ─────────────────────────────────────────────────────────────────────


def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF r[1..max_lag]; x is 1D float32."""
    n = x.size
    # mean-center (detrended is near zero-mean, but do it anyway)
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return np.zeros(max_lag, dtype=np.float32)
    r = np.empty(max_lag, dtype=np.float32)
    # O(nK) naive is fine for K<=20
    for k in range(1, max_lag + 1):
        num = np.dot(x[k:], x[:-k])
        r[k - 1] = num / ((n - k) * var)  # consistent scaling across lags
    return r


def pacf_yw(x: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF via Yule–Walker / Durbin–Levinson; returns pacf[1..max_lag]."""
    r = acf_1d(x, max_lag)  # r[1..K]
    r0 = 1.0
    # build autocorr sequence r_full[0..K]
    r_full = np.concatenate(([r0], r))
    pacf = np.zeros(max_lag, dtype=np.float32)
    # Durbin–Levinson
    phi = np.zeros((max_lag + 1, max_lag + 1), dtype=np.float32)
    sig = np.empty(max_lag + 1, dtype=np.float32)
    phi[1, 1] = r_full[1]
    sig[1] = 1 - r_full[1] ** 2
    pacf[0] = phi[1, 1]
    for k in range(2, max_lag + 1):
        num = r_full[k] - np.dot(phi[1:k, k - 1], r_full[1:k][::-1])
        den = sig[k - 1] if sig[k - 1] > 0 else EPS
        phi[k, k] = num / den
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)
        pacf[k - 1] = phi[k, k]
    return pacf


def ljung_box_z(x: np.ndarray, m: int) -> float:
    """Length/m-invariant LBQ z-score using unbiased ACF and common m."""
    n = x.size
    r = acf_1d(x.astype(np.float32), m)  # unbiased ACF (your fixed version)
    ks = np.arange(1, m + 1, dtype=np.float32)
    Q = n * (n + 2.0) * np.sum((r**2) / (n - ks))
    z = (Q - m) / np.sqrt(2.0 * m)
    return float(0.0 if not np.isfinite(z) else z)


def compute_autocorr_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    acf_max_lag: int = ACF_MAX_LAG,  # ACF/PACF lags 1..acf_max_lag for shape/summaries
    lbq_m: int = LBQ_M,  # Ljung–Box Q statistic uses lbq_m
) -> pd.DataFrame:
    """
    Autocorrelation features on the *detrended* standardized series.

    Outputs (one row per id):
      - acf1_delta, acf2_delta                   : ACF lag-1/2 (after − before)
      - pacf1_delta, pacf2_delta                 : PACF lag-1/2 (after − before)
      - short_lag_acf_dep_delta                  : Σ_{ℓ=1..K} |ρ(ℓ)|/ℓ  (after − before)
      - lbq_stat_delta                           : Ljung–Box Q_m (after − before), m = m_lbq
      - alt_signed_sum_delta                     : (Σ_{ℓ=1..K} (−1)^{ℓ−1} ρ_a(ℓ)) − same_before

    Notes:
      • Crashes if a segment contains non-finite values (enforce data hygiene).
      • Fast O(nK) per id (K small), no heavy FFT needed.
    """
    # Caching
    prefix = "autocorr"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # Group series
    zcol = "detrended"
    zd_b = _group_series(X_prep, zcol, 0)
    zd_a = _group_series(X_prep, zcol, 1)

    # Materialize arrays once (NumPy, float32 for stability)
    zd_b = zd_b.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    zd_a = zd_a.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    ids = zd_b.index

    # ---------- compute per id ----------
    acf1_d, acf2_d = [], []
    pacf2_d = []
    shortlag_d = []
    lbq_d = []
    alt_sum_d = []

    for i in ids:
        xb = zd_b.loc[i].astype(np.float32, copy=False)
        xa = zd_a.loc[i].astype(np.float32, copy=False)

        # ACF/PACF up to K
        acf_b = acf_1d(xb, acf_max_lag)  # r[1..acf_max_lag]
        acf_a = acf_1d(xa, acf_max_lag)
        pacf_b = pacf_yw(xb, acf_max_lag)
        pacf_a = pacf_yw(xa, acf_max_lag)

        # lag-1/2 deltas
        acf1_d.append(acf_a[0] - acf_b[0] if acf_max_lag >= 1 else 0.0)
        acf2_d.append(acf_a[1] - acf_b[1] if acf_max_lag >= 2 else 0.0)
        pacf2_d.append(pacf_a[1] - pacf_b[1] if acf_max_lag >= 2 else 0.0)

        # short-lag dependence Σ |ρ(ℓ)|/ℓ, starting from l=2
        weights = 1.0 / np.arange(2, acf_max_lag + 1, dtype=np.float32)
        s_b = np.sum(np.abs(acf_b[1:]) * weights)
        s_a = np.sum(np.abs(acf_a[1:]) * weights)
        shortlag_d.append(s_a - s_b)

        # Ljung–Box Q (m = lbq_m)
        lbq_d.append(ljung_box_z(xa, lbq_m) - ljung_box_z(xb, lbq_m))

        # Alternative signed sum Σ (-1)^{ℓ-1} ρ(ℓ)
        signs = np.where((np.arange(1, acf_max_lag + 1) % 2) == 1, 1.0, -1.0)
        alt_b = float(np.sum(signs * acf_b))
        alt_a = float(np.sum(signs * acf_a))
        alt_sum_d.append(alt_a - alt_b)

    autoc = pd.DataFrame(
        {
            "acf1_delta": acf1_d,
            "acf2_delta": acf2_d,
            "pacf2_delta": pacf2_d,
            "shortlag_l1_delta": shortlag_d,
            "lbq_stat_delta": lbq_d,
            "alt_signed_sum_delta": alt_sum_d,
        },
        index=ids,
        dtype=np.float32,
    )

    if not inference:
        _save_cache(autoc, prefix)
    return autoc


# ─────────────────────────────────────────────────────────────────────
# TESTS & DISTANCES BLOCK
# ─────────────────────────────────────────────────────────────────────


def _idr(x):
    q = np.quantile(x, [0.25, 0.75])
    return float(q[1] - q[0])


def _ks_normalized(before: np.ndarray, after: np.ndarray) -> float:
    """
    Kolmogorov two-sample normalized statistic:
      KS_norm = (sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * D,
    where n_eff = n0*n1/(n0+n1) and D is the sup ECDF distance.
    Returns 0.0 if either segment is empty.
    """
    n0, n1 = before.size, after.size
    if n0 == 0 or n1 == 0:
        return 0.0

    # compute D (two-sample KS) inline (merge-walk)
    xa = np.sort(before)
    xb = np.sort(after)
    ia = ib = 0
    cdfa = cdfb = 0.0
    D = 0.0
    while ia < n0 and ib < n1:
        if xa[ia] <= xb[ib]:
            ia += 1
            cdfa = ia / n0
        else:
            ib += 1
            cdfb = ib / n1
        D = max(D, abs(cdfa - cdfb))
    if ia < n0:
        D = max(D, abs(1.0 - cdfb))
    if ib < n1:
        D = max(D, abs(cdfa - 1.0))

    n_eff = (n0 * n1) / (n0 + n1)
    s = np.sqrt(max(n_eff, 1.0))
    return float((s + 0.12 + 0.11 / s) * D)


def _css_normalized(before: np.ndarray, after: np.ndarray) -> float:
    """
    Inclán–Tiao CUSUM-of-squares with Brownian-bridge scaling:
      CSS_norm = sqrt(n) * max_t | S_t / S_T - t/T |,
    computed on the concatenated series (before || after).
    """
    if before.size + after.size <= 1:
        return 0.0
    y = np.concatenate([before, after])
    n = y.size
    s2 = np.cumsum(y * y)
    ST = s2[-1]
    if ST <= 0:
        return 0.0
    t = np.arange(1, n + 1, dtype=np.float32)
    D = s2 / ST - t / n
    css = float(np.max(np.abs(D)))
    return float(np.sqrt(n) * css)


def _wasserstein_quant(a, b, qs, scale):
    Qa = np.quantile(a, qs)
    Qb = np.quantile(b, qs)
    w1 = np.mean(np.abs(Qa - Qb))
    denom = max(scale, 1e-8)
    return float(w1 / denom)


def _js_divergence(a, b, q_edges):
    # Quantile-based bin edges from BEFORE segment
    edges = np.quantile(a, q_edges)
    # ensure strictly increasing edges (collapse-safe)
    edges = np.unique(edges)
    # histograms
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    # Jeffreys smoothing (0.5) → probabilities
    pa = pa.astype(np.float32) + 0.5
    pb = pb.astype(np.float32) + 0.5
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    # JS in nats
    with np.errstate(divide="ignore", invalid="ignore"):
        KL_am = np.sum(pa * (np.log(pa) - np.log(m)))
        KL_bm = np.sum(pb * (np.log(pb) - np.log(m)))
    JS = 0.5 * (KL_am + KL_bm)
    if not np.isfinite(JS):
        return 0.0
    return float(JS)


def _mmd2_rbf(a, b, sigma, mmd_max_n):
    # Unbiased MMD^2; cap sample sizes for speed
    na = a.size
    nb = b.size
    if na > mmd_max_n:
        idx = np.linspace(0, na - 1, mmd_max_n, dtype=int)
        a = a[idx]
        na = a.size
    if nb > mmd_max_n:
        idx = np.linspace(0, nb - 1, mmd_max_n, dtype=int)
        b = b[idx]
        nb = b.size
    gamma = 1.0 / (2.0 * sigma * sigma)

    def _kxx(x):
        # exclude diagonal for unbiased estimator
        d2 = (x[:, None] - x[None, :]) ** 2
        np.fill_diagonal(d2, 0.0)
        K = np.exp(-gamma * d2)
        return K.sum() / (x.size * (x.size - 1) + EPS)

    def _kxy(x, y):
        d2 = (x[:, None] - y[None, :]) ** 2
        K = np.exp(-gamma * d2)
        return K.mean()

    kxx = _kxx(a) if na > 1 else 0.0
    kyy = _kxx(b) if nb > 1 else 0.0
    kxy = _kxy(a, b)
    return float(kxx + kyy - 2.0 * kxy)


def _gaussian_glr_per(before: np.ndarray, after: np.ndarray) -> float:
    """
    Per-sample Gaussian GLR for joint mean+variance change (length-invariant).
    LRT = n*log(s2_pooled) - n0*log(s2_b) - n1*log(s2_a); return LRT / n.
    s2_* are MLE variances about their own means (ddof=0).
    """
    n0 = before.size
    n1 = after.size
    n = n0 + n1
    s2_b = float(np.var(before, ddof=0))
    s2_a = float(np.var(after, ddof=0))
    # guard against degenerate variance
    s2_b = max(s2_b, EPS)
    s2_a = max(s2_a, EPS)
    s2_p = (n0 * s2_b + n1 * s2_a) / n
    s2_p = max(s2_p, EPS)
    lrt = n * np.log(s2_p) - n0 * np.log(s2_b) - n1 * np.log(s2_a)
    return float(lrt / n)


def _arch_neglogp(x: np.ndarray, L: int) -> float:
    """
    ARCH-LM: compute LM = n*R^2, then -log p with df=L (p from chi-square upper tail).
    Uses SciPy's gammaincc (regularized upper incomplete gamma).
    """
    # You already have _arch_lm_LM(x, L) elsewhere; if not, substitute here.
    LM = _arch_lm_LM(x, L=L)
    # p = P[Chi2_L >= LM] = gammaincc(L/2, LM/2)
    p = float(gammaincc(0.5 * L, 0.5 * max(LM, 0.0)))
    return float(-np.log(max(p, EPS)))


def compute_tests_distances_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    js_quantile_bins: np.ndarray = JS_QUANTILE_BINS,  # e.g., np.linspace(0,1,33)
    mmd_max_n: int = MMD_MAX_N,  # e.g., 512
    arch_L: int = ARCH_L,  # e.g., 5
) -> pd.DataFrame:
    """
    Length-stable tests & distances between BEFORE and AFTER on standardized z.

    Tests (length-invariant forms):
      - gauss_glr_per_sample      : per-sample Gaussian GLR (mean+variance)
      - css_norm                  : √n * CSS on concatenated series
      - archlm_neglogp_delta      : (-log p)_after - (-log p)_before   [ARCH-LM, df=arch_L]

    Distances (length-agnostic):
      - ks_norm                   : Kolmogorov D with Massey normalization
      - js_divergence             : quantile-binned JS with Jeffreys smoothing
      - mmd2_rbf_idrband_equal    : unbiased MMD² (RBF), equal-size subsample, σ = IDR_before/2
    """
    prefix = "tests_distances"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # standardized segments
    z_b = _group_series(X_prep, "standardized", 0).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    z_a = _group_series(X_prep, "standardized", 1).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    ids = z_b.index

    # accumulators
    glr_per_L, css_norm_L, arch_neglogp_delta_L = [], [], []
    ks_norm_L, js_div_L, mmd2_equal_L = [], [], []

    for i in ids:
        b = z_b.loc[i]
        a = z_a.loc[i]

        # --- Tests (normalized) ---
        glr_per = _gaussian_glr_per(b, a)
        css_norm = _css_normalized(b, a)

        arch_b = _arch_neglogp(b, arch_L)
        arch_a = _arch_neglogp(a, arch_L)
        arch_delta = float(arch_a - arch_b)

        # --- Distances (no caps) ---
        ks_norm = _ks_normalized(b, a)
        js = _js_divergence(b, a, js_quantile_bins)

        # MMD² with equal-size subsampling (deterministic linspace indices)
        m = int(min(b.size, a.size, mmd_max_n))
        if m >= 2:
            idx_b = np.linspace(0, b.size - 1, m, dtype=int)
            idx_a = np.linspace(0, a.size - 1, m, dtype=int)
            bb = b[idx_b]
            aa = a[idx_a]
            sigma = max(_idr(b), EPS) / 2.0
            mmd2 = _mmd2_rbf(
                bb, aa, sigma, mmd_max_n=m
            )  # mmd_max_n=m since already subsampled equally
        else:
            mmd2 = 0.0

        # collect
        glr_per_L.append(glr_per)
        css_norm_L.append(css_norm)
        arch_neglogp_delta_L.append(arch_delta)
        ks_norm_L.append(ks_norm)
        js_div_L.append(js)
        mmd2_equal_L.append(mmd2)

    out = pd.DataFrame(
        {
            "gauss_glr_per_sample": glr_per_L,
            "css_norm": css_norm_L,
            "archlm_neglogp_delta": arch_neglogp_delta_L,
            "ks_norm": ks_norm_L,
            "js_divergence": js_div_L,
            "mmd2_rbf_idrband_equal": mmd2_equal_L,
        },
        index=ids,
        dtype=np.float32,
    )

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# FREQUENCY BLOCK
# ─────────────────────────────────────────────────────────────────────


def _psd_rfft(x: np.ndarray):
    """Return (freq in [0,0.5], power spectrum). Mean-centered; no window for speed."""
    n = x.size
    y = x - x.mean()
    fft = np.fft.rfft(y)
    P = (fft.real**2 + fft.imag**2) / max(n, 1)
    f = np.fft.rfftfreq(n, d=1.0)  # normalized to sampling step 1 → Nyquist 0.5
    return f, P


def _spectral_centroid(f, P):
    tot = P.sum()
    return float((f * P).sum() / (tot + EPS)) if tot > 0 else 0.0


def _log_flatness(P):
    """log( geometric_mean / arithmetic_mean )."""
    Pp = P + EPS
    return float(np.exp(np.mean(np.log(Pp))) / np.mean(Pp) + 0.0)  # flatness in (0,1]
    # we’ll convert to log-domain delta below


def _bandpower_logratio(f, P, bands):
    tot = P.sum()
    if tot <= 0:
        return [0.0] * len(bands)
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        frac = P[m].sum() / (tot + EPS)
        out.append(frac)
    return out  # we’ll take log-ratio a/b later


def _dwt_l3_ratio(x, dwt_wavelet, dwt_level):
    if not x.flags.writeable:
        x = x.copy()
    coeffs = pywt.wavedec(x, dwt_wavelet, level=dwt_level, mode="symmetric")
    # coeffs = [cA_L, cD_L, ..., cD_1]
    details = coeffs[1:]  # list of arrays cD_L..cD_1
    if len(details) < 3:
        return 0.0
    cD3 = details[-3]  # level-3 detail
    e3 = float(np.sum(cD3 * cD3))
    etot = float(sum(np.sum(d * d) for d in details)) + EPS
    return e3 / etot


def _perm_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    Normalized permutation entropy in [0,1] using ordinal patterns.
    Deterministic tie-breaking via stable argsort.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    span = (m - 1) * tau
    N = n - span
    if m < 2 or tau < 1 or N <= 0:
        return 0.0
    # counts
    counts = {}
    # stable tie-breaking: argsort twice to get ranks
    for i in range(N):
        window = x[i : i + span + 1 : tau]
        # rank vector (0..m-1) with stable tie handling
        order = np.argsort(window, kind="mergesort")
        ranks = np.empty(m, dtype=np.int64)
        ranks[order] = np.arange(m, dtype=np.int64)
        key = tuple(ranks.tolist())
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=np.float64)
    H = -np.sum(probs * np.log(probs + EPS))
    Hmax = np.log(math.factorial(m))
    return float(H / (Hmax + EPS))


def compute_frequency_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    bands: tuple = FREQ_BANDS,  # normalized freq bands
    dwt_wavelet: str = DWT_WAVELET,
    dwt_level: int = DWT_LEVEL,
    entropy_m1: int = ENTROPY_M1,
    entropy_m2: int = ENTROPY_M2,
    entropy_tau: int = ENTROPY_TAU,
) -> pd.DataFrame:
    """
    Frequency & entropy features per id.

    Spectral (on detrended):
      - spectral centroid (delta)
      - spectral flatness (delta of log-flatness)
      - band power fractions over 'bands' (log-ratio after/before per band)
      - DWT level-3 detail energy / total detail energy (log-ratio after/before)
    """
    prefix = "frequency"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # --- choose series ---
    z_std = "standardized"
    z_det = "detrended"

    z_b = _group_series(X_prep, z_std, 0)
    z_a = _group_series(X_prep, z_std, 1)
    zd_b = _group_series(X_prep, z_det, 0)
    zd_a = _group_series(X_prep, z_det, 1)

    z_b = z_b.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    z_a = z_a.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    zd_b = zd_b.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    zd_a = zd_a.apply(lambda s: s.to_numpy(dtype=np.float32, copy=False))
    ids = zd_b.index

    # ---- accumulators ----
    log_flatness_d = []
    band_logratios = [[] for _ in bands]  # list of lists
    dwt_l3_logratio = []
    perm_m3_delta = []
    perm_m5_delta = []

    # ---- compute per id ----
    for i in ids:
        xb = zd_b.loc[i]
        xa = zd_a.loc[i]

        # FFT stats (detrended)
        fb, Pb = _psd_rfft(xb)
        fa, Pa = _psd_rfft(xa)

        # Spectral flatness (use log-flatness then delta)
        sf_b = _log_flatness(Pb)
        sf_a = _log_flatness(Pa)
        # store as log of flatness (so delta is log-ratio); equivalently take log here
        log_flatness_d.append(np.log(sf_a + EPS) - np.log(sf_b + EPS))

        # Band powers → fractions → log-ratios
        frac_b = _bandpower_logratio(fb, Pb, bands)
        frac_a = _bandpower_logratio(fa, Pa, bands)
        for bi, (fb_i, fa_i) in enumerate(zip(frac_b, frac_a)):
            band_logratios[bi].append(np.log((fa_i + EPS) / (fb_i + EPS)))

        # DWT L3 energy ratio (detrended) → log-ratio
        r_b = _dwt_l3_ratio(xb, dwt_wavelet, dwt_level)
        r_a = _dwt_l3_ratio(xa, dwt_wavelet, dwt_level)
        dwt_l3_logratio.append(np.log((r_a + EPS) / (r_b + EPS)))

        # permutation entropy deltas (m=3 and m=5, tau=1)
        zb = z_b.loc[i]
        za = z_a.loc[i]
        pe_b_m3 = _perm_entropy(zb, m=entropy_m1, tau=entropy_tau)
        pe_a_m3 = _perm_entropy(za, m=entropy_m1, tau=entropy_tau)
        pe_b_m5 = _perm_entropy(zb, m=entropy_m2, tau=entropy_tau)
        pe_a_m5 = _perm_entropy(za, m=entropy_m2, tau=entropy_tau)
        perm_m3_delta.append(pe_a_m3 - pe_b_m3)
        perm_m5_delta.append(pe_a_m5 - pe_b_m5)

    # ---- assemble ----
    cols = {
        "spec_flatness_logratio": log_flatness_d,
        "dwt_l3_energy_logratio": dwt_l3_logratio,
        "perm_entropy_m3_delta": perm_m3_delta,
        "perm_entropy_m5_delta": perm_m5_delta,
    }
    # add band logratios with names
    for bi, _ in enumerate(bands):
        cols[f"bandpower_b{bi + 1}_logratio"] = band_logratios[bi]

    out = pd.DataFrame(cols, index=ids, dtype=np.float32)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# DIFFERENCES BLOCK
# ─────────────────────────────────────────────────────────────────────


def _shortlag_L1(x: np.ndarray, K: int) -> float:
    r = acf_1d(x.astype(np.float32), K)  # r[1..K]
    w = 1.0 / np.arange(2, K + 1, dtype=np.float32)
    return float(np.sum(np.abs(r[1:]) * w))


def compute_differences_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    quantile_fine_grid: list[float] = QUANTILE_FINE_GRID,
    crossing_rate_deadband: float = CROSSING_RATE_DEADBAND,
    lbq_m: int = LBQ_M,
    acf_max_lag: int = ACF_MAX_LAG,
) -> pd.DataFrame:
    """
    Returns per-id features (float32):
      - diff_mad_logratio                   : log(MAD(d))_after − log(MAD(d))_before
      - diff_w1_scaled                      : Wasserstein-1(d_a,d_b) via quantile grid / IDR_before
      - diff_signflips_logit_delta          : logit(sign-flip rate)_after − logit(...)_before
      - diff_shortlag_l1_delta              : Σ_{ℓ=1..K} |ACF_dm(ℓ)| / ℓ  (delta)
      - diff_lbq_stat_delta                 : Ljung–Box Q(m) (delta)
      - diff_spec_centroid_delta            : spectral centroid (delta)
      - absdiff_up_excd_q95_logit_delta     : logit P(a > T95_before) (delta)
      - absdiff_w1_scaled                   : Wasserstein-1(a_a,a_b) via quantile grid / IDR_before
      - absdiff_acf1_delta                  : lag-1 ACF(am) (delta)
    """
    prefix = "differences"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # keep only needed cols to speed up groupby.apply
    cols = [
        "period",
        "diff_standardized",
        "diff_detrended",
        "absdiff_detrended",
    ]
    df = X_prep[cols].copy()

    # ---------- per-id computation ----------
    def _one_id(g: pd.DataFrame) -> pd.Series:
        gb = g[g["period"] == 0]
        ga = g[g["period"] == 1]

        # fetch arrays
        d_b = gb["diff_standardized"].to_numpy(np.float32, copy=False)
        d_a = ga["diff_standardized"].to_numpy(np.float32, copy=False)
        dm_b = gb["diff_detrended"].to_numpy(np.float32, copy=False)
        dm_a = ga["diff_detrended"].to_numpy(np.float32, copy=False)
        am_b = gb["absdiff_detrended"].to_numpy(np.float32, copy=False)
        am_a = ga["absdiff_detrended"].to_numpy(np.float32, copy=False)

        # Median-crossing rate change (deadband) on standardized; Q=0
        _, _, cr_b = crossing_rates_logits(d_b, Q=0.0, eps=crossing_rate_deadband)
        _, _, cr_a = crossing_rates_logits(d_a, Q=0.0, eps=crossing_rate_deadband)
        signflips_logit_delta = float(cr_a - cr_b)

        # Extreme events (Jeffreys-smoothed logits)
        qs = quantile_fine_grid
        Qb = np.quantile(d_b, qs)
        Qb5, Qb95 = Qb[0], Qb[8]
        _, b2 = exceedance_logits(d_b, k_abs=Qb95, k_fixed=3.0)
        _, a2 = exceedance_logits(d_a, k_abs=Qb95, k_fixed=3.0)

        u_b, l_b = upper_lower_logits(d_b, q_low=Qb5, q_high=Qb95)
        u_a, l_a = upper_lower_logits(d_a, q_low=Qb5, q_high=Qb95)

        # --- On dm: acf1_delta, short-lag L1, Ljung–Box Q, spectral centroid (all deltas) ---
        diff_shortlag_l1_delta = _shortlag_L1(dm_a, acf_max_lag) - _shortlag_L1(
            dm_b, acf_max_lag
        )
        diff_lbq_delta = ljung_box_z(dm_a, lbq_m) - ljung_box_z(dm_b, lbq_m)

        # Spectral centroid
        fa, Pa = _psd_rfft(dm_a)
        fb, Pb = _psd_rfft(dm_b)
        diff_spec_centroid_delta = _spectral_centroid(fa, Pa) - _spectral_centroid(
            fb, Pb
        )

        # --- On am: volatility clustering → lag-1 ACF delta ---
        r1_b = acf_1d(am_b.astype(np.float32), 1)[0]
        r1_a = acf_1d(am_a.astype(np.float32), 1)[0]
        absdiff_acf1_delta = float(r1_a - r1_b)

        return pd.Series(
            {
                "diff_abs_excd_3_logit_delta": a2 - b2,
                "diff_up_excd_q95_logit_delta": u_a - u_b,
                "diff_low_excd_q5_logit_delta": l_a - l_b,
                "diff_signflips_logit_delta": signflips_logit_delta,
                "diff_shortlag_l1_delta": diff_shortlag_l1_delta,
                "diff_lbq_stat_delta": diff_lbq_delta,
                "diff_spec_centroid_delta": diff_spec_centroid_delta,
                "absdiff_acf1_delta": absdiff_acf1_delta,
            },
            dtype=np.float32,
        )

    out = (
        df.groupby(level="id", sort=False, group_keys=False)
        .apply(_one_id)
        .astype(np.float32)
    )

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# ABSOLUTE BLOCK
# ─────────────────────────────────────────────────────────────────────


def compute_absolute_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    acf_max_lag: int = ACF_MAX_LAG,
    lbq_m: int = LBQ_M,
) -> pd.DataFrame:
    """
    Absolute magnitudes (on 'absval_detrended'):
      - absval_acf1_delta       : lag-1 ACF (after − before)
      - absval_lbq_stat_delta   : Ljung–Box Q(m) (after − before), m=20 by default
    """
    prefix = "absolute"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    col = "absval_detrended"
    df = X_prep[[col, "period"]].copy()

    def _one_id(g: pd.DataFrame) -> pd.Series:
        gb = g[g["period"] == 0][col].to_numpy(np.float32, copy=False)
        ga = g[g["period"] == 1][col].to_numpy(np.float32, copy=False)

        # lag-1 ACF
        r1_b = acf_1d(gb, 1)[0]
        r1_a = acf_1d(ga, 1)[0]
        acf1_delta = float(r1_a - r1_b)

        # Shortlag L1
        shortlag_l1_delta = _shortlag_L1(ga, acf_max_lag) - _shortlag_L1(
            gb, acf_max_lag
        )

        # LBQ stat delta
        lbq_delta = ljung_box_z(ga, lbq_m) - ljung_box_z(gb, lbq_m)

        return pd.Series(
            {
                "absval_acf1_delta": acf1_delta,
                "absval_shortlag_L1_delta": shortlag_l1_delta,
                "absval_lbq_stat_delta": lbq_delta,
            },
            dtype=np.float32,
        )

    out = (
        df.groupby(level="id", sort=False, group_keys=False)
        .apply(_one_id)
        .astype(np.float32)
    )

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# SQUARED BLOCK
# ─────────────────────────────────────────────────────────────────────


def compute_squared_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    acf_max_lag: int = ACF_MAX_LAG,
    lbq_m: int = LBQ_M,
) -> pd.DataFrame:
    """
    Absolute magnitudes (on 'squared_detrended'):
      - squared_acf1_delta       : lag-1 ACF (after − before)
      -
      - squared_lbq_stat_delta   : Ljung–Box Q(m) (after − before), m=20 by default
    """
    prefix = "squared"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    col = "squared_detrended"
    df = X_prep[[col, "period"]].copy()

    def _one_id(g: pd.DataFrame) -> pd.Series:
        gb = g[g["period"] == 0][col].to_numpy(np.float32, copy=False)
        ga = g[g["period"] == 1][col].to_numpy(np.float32, copy=False)

        # lag-1 ACF
        r1_b = acf_1d(gb, 1)[0]
        r1_a = acf_1d(ga, 1)[0]
        acf1_delta = float(r1_a - r1_b)

        # Shortlag L1
        shortlag_l1_delta = _shortlag_L1(ga, acf_max_lag) - _shortlag_L1(
            gb, acf_max_lag
        )

        # LBQ stat delta
        lbq_delta = ljung_box_z(ga, lbq_m) - ljung_box_z(gb, lbq_m)

        return pd.Series(
            {
                "squared_acf1_delta": acf1_delta,
                "squared_shortlag_L1_delta": shortlag_l1_delta,
                "squared_lbq_stat_delta": lbq_delta,
            },
            dtype=np.float32,
        )

    out = (
        df.groupby(level="id", sort=False, group_keys=False)
        .apply(_one_id)
        .astype(np.float32)
    )

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# BOUNDARY LOCAL BLOCK
# ─────────────────────────────────────────────────────────────────────


def _window_around_boundary(
    arr_before: np.ndarray,
    arr_after: np.ndarray,
    w_before: int,
    w_after: int,
    skip_after: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return boundary-local windows:
      before: last w_before samples,
      after : first w_after samples (optionally skip 'skip_after' transient).
    """
    if arr_before.size == 0 or arr_after.size == 0:
        return arr_before, arr_after
    wb = int(min(w_before, arr_before.size))
    sa = int(min(skip_after, max(0, arr_after.size - 1)))
    wa = int(min(w_after, max(0, arr_after.size - sa)))
    b = arr_before[-wb:] if wb > 0 else arr_before[:0]
    a = arr_after[sa : sa + wa] if wa > 0 else arr_after[:0]
    return b.astype(np.float32, copy=False), a.astype(np.float32, copy=False)


def _ecdf_logit_against_before(v: float, before_sorted: np.ndarray) -> float:
    """Jeffreys-smoothed logit rank of value v against BEFORE sample."""
    n = before_sorted.size
    lt = np.searchsorted(before_sorted, v, side="left")
    le = np.searchsorted(before_sorted, v, side="right")
    rank = lt + 0.5 * (le - lt)
    p = (rank + 0.5) / (n + 1.0)
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.log(p / (1.0 - p)))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta in [-1, 1]: P(a>b) - P(a<b).
    O(n log n) using searchsorted counts.
    """
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return 0.0
    b_sorted = np.sort(b)
    # counts for each a: how many b are < a, and > a
    lt_counts = np.searchsorted(b_sorted, a, side="left")  # b < a
    gt_counts = nb - np.searchsorted(b_sorted, a, side="right")  # b > a
    gt = int(np.sum(lt_counts))  # P(a>b) numerator
    lt = int(np.sum(gt_counts))  # P(a<b) numerator
    return float((gt - lt) / (na * nb))


def _chow_F(y_b: np.ndarray, y_a: np.ndarray) -> float:
    """
    Chow F test at the boundary for y ~ a + b t.
    Returns the F-statistic. Uses clipped/detrended to limit outliers.
    """

    def _ols_ssr(y):
        n = y.size
        if n < 2:
            return 0.0, 0
        t = np.arange(n, dtype=np.float32)
        X = np.c_[np.ones(n), t]
        # OLS via normal equations (2x2)
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        resid = y - X @ beta
        return float(np.dot(resid, resid)), 2  # ssr, k

    ssr_b, k = _ols_ssr(y_b)
    ssr_a, _ = _ols_ssr(y_a)
    y = np.concatenate([y_b, y_a])
    ssr_pooled, _ = _ols_ssr(y)
    n_b, n_a = max(0, y_b.size), max(0, y_a.size)
    n = n_b + n_a
    # F = ((SSR_pooled - (SSR_b+SSR_a)) / k) / ((SSR_b+SSR_a) / (n - 2k))
    num = (ssr_pooled - (ssr_b + ssr_a)) / max(k, 1)
    den = (ssr_b + ssr_a) / max(n - 2 * k, 1)
    F = num / den if den > 0 else 0.0
    return float(max(F, 0.0)) if np.isfinite(F) else 0.0


def _cusum_signed_stat(y_b: np.ndarray, y_a: np.ndarray) -> float:
    """
    Signed CUSUM across boundary: build combined series (mean-centered),
    compute cumulative sum; report jump as end_after_cusum - last_before_cusum,
    normalized by std * sqrt(n) for rough scale invariance.
    """
    y_b = y_b.astype(np.float32)
    y_a = y_a.astype(np.float32)
    if y_b.size + y_a.size < 3:
        return 0.0
    y = np.concatenate([y_b, y_a])
    y = y - y.mean()
    s = np.cumsum(y)
    j = y_b.size
    num = s[-1] - s[j - 1] if j > 0 else s[-1]
    denom = (y.std(ddof=0) + EPS) * np.sqrt(y.size)
    return float(num / denom)


def _gaussian_glr(zb: np.ndarray, za: np.ndarray) -> float:
    """
    Gaussian log-likelihood ratio at the given split:
      LRT = n*log(var_all) - n0*log(var_b) - n1*log(var_a)
    Uses MLE variances (ddof=0) about each segment's own mean.
    """
    n0 = int(zb.size)
    n1 = int(za.size)
    n = n0 + n1
    if n0 <= 1 or n1 <= 1:
        return 0.0
    m0 = float(np.mean(zb))
    v0 = float(np.mean((zb - m0) ** 2))
    m1 = float(np.mean(za))
    v1 = float(np.mean((za - m1) ** 2))
    y = np.concatenate([zb, za], axis=0)
    m = float(np.mean(y))
    v = float(np.mean((y - m) ** 2))
    v0 = max(v0, EPS)
    v1 = max(v1, EPS)
    v = max(v, EPS)
    return float(n * np.log(v) - n0 * np.log(v0) - n1 * np.log(v1))


def _arch_lm_LM(z: np.ndarray, L: int = 5) -> float:
    """
    Engle's ARCH LM statistic on a single segment z (standardized):
      1) center z, set e2 = (z - mean(z))^2
      2) regress e2[L:] on [1, e2_{t-1},...,e2_{t-L}]
      3) LM = n_eff * R^2
    """
    z = z.astype(np.float64)
    n = z.size
    if n <= L + 1:
        return 0.0
    e2 = (z - z.mean()) ** 2
    Y = e2[L:]
    Xcols = [np.ones_like(Y)]
    for j in range(1, L + 1):
        Xcols.append(e2[L - j : n - j])
    X = np.column_stack(Xcols)  # shape (n_eff, 1+L)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Yhat = X @ beta
    ssr = float(np.sum((Yhat - Y.mean()) ** 2))  # regression sum of squares
    sst = float(np.sum((Y - Y.mean()) ** 2)) + EPS  # total sum of squares
    R2 = ssr / sst
    n_eff = Y.shape[0]
    return float(n_eff * R2)


def compute_boundary_local_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    w_local: list[int] = BOUND_WINDOW_SIZES,
    skip_after: int = BOUND_SKIP_AFTER,
    acf_K: int = BOUND_ACF_MAX_LAG,
    eps_deadband: float = CROSSING_RATE_DEADBAND,
    quantile_coarse_grid: list[float] = QUANTILE_COARSE_GRID,
) -> pd.DataFrame:
    """
    Boundary-local features (windowed around the split): jumps, local scale/trend,
    crossings/residence, short-lag ACF/L1, spectral centroid on detrended, local W1,
    RMS logratio, diff median jump, Cliff's delta, Chow F, signed CUSUM.

    If `w_local` is a list (e.g., [32, 64, 128]), compute the full feature set
    for each window size and suffix columns with `_w{size}`.
    """
    w_list = sorted(set(int(w) for w in w_local))

    prefix = "boundary_local"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    require_cols = [
        "period",
        "standardized",
        "clipped",
        "detrended",
        "diff_standardized",
        "diff2_standardized",
        "diff2_detrended",
    ]
    g = X_prep[require_cols].groupby(level="id", sort=False)

    def _one_id(df: pd.DataFrame, w_curr: int) -> pd.Series:
        b = df[df["period"] == 0]
        a = df[df["period"] == 1]

        z_b = b["standardized"].to_numpy(np.float32, copy=False)
        z_a = a["standardized"].to_numpy(np.float32, copy=False)
        zc_b = b["clipped"].to_numpy(np.float32, copy=False)
        zc_a = a["clipped"].to_numpy(np.float32, copy=False)
        zd_b = b["detrended"].to_numpy(np.float32, copy=False)
        zd_a = a["detrended"].to_numpy(np.float32, copy=False)
        dz_b = b["diff_standardized"].to_numpy(np.float32, copy=False)
        dz_a = a["diff_standardized"].to_numpy(np.float32, copy=False)
        dd_b = b["diff2_standardized"].to_numpy(np.float32, copy=False)
        dd_a = a["diff2_standardized"].to_numpy(np.float32, copy=False)
        ddm_b = b["diff2_detrended"].to_numpy(np.float32, copy=False)
        ddm_a = a["diff2_detrended"].to_numpy(np.float32, copy=False)

        # Local windows (use current w)
        z_bw, z_aw = _window_around_boundary(z_b, z_a, w_curr, w_curr, skip_after)
        zc_bw, zc_aw = _window_around_boundary(zc_b, zc_a, w_curr, w_curr, skip_after)
        zd_bw, zd_aw = _window_around_boundary(zd_b, zd_a, w_curr, w_curr, skip_after)
        dz_bw, dz_aw = _window_around_boundary(dz_b, dz_a, w_curr, w_curr, skip_after)
        dd_bw, dd_aw = _window_around_boundary(dd_b, dd_a, w_curr, w_curr, skip_after)
        ddmw, ddaw = _window_around_boundary(ddm_b, ddm_a, w_curr, w_curr, skip_after)

        # Local median jump
        local_median_jump = (
            float(np.median(z_aw) - np.median(z_bw))
            if (z_aw.size and z_bw.size)
            else 0.0
        )

        # Local scale logratio (MAD)
        mad_b = 1.4826 * _mad(z_bw) + EPS
        mad_a = 1.4826 * _mad(z_aw) + EPS
        local_scale_logratio = float(np.log(mad_a) - np.log(mad_b))

        # Local slope jump on detrended
        slope_jump = _ols_slope(zd_aw) - _ols_slope(zd_bw)

        # Local IDR logratio (Q10..Q90 on standardized)
        qs = quantile_coarse_grid
        Qb = np.quantile(z_bw, qs) if z_bw.size else np.zeros(len(qs), dtype=np.float32)
        Qa = np.quantile(z_aw, qs) if z_aw.size else np.zeros(len(qs), dtype=np.float32)
        Q10_b, Q90_b = Qb[0], Qb[-1]
        Q10_a, Q90_a = Qa[0], Qa[-1]
        idr_b = Q90_b - Q10_b + EPS
        idr_a = Q90_a - Q10_a + EPS
        IDR_logratio = float(np.log(idr_a) - np.log(idr_b))

        # Exceedances using BEFORE thresholds
        _, Qb95 = np.quantile(z_b, [0.05, 0.95]) if z_b.size else (0.0, 0.0)
        b1, b2 = exceedance_logits(z_bw, k_abs=Qb95, k_fixed=3.0)
        a1, a2 = exceedance_logits(z_aw, k_abs=Qb95, k_fixed=3.0)

        # Crossing rates & asymmetry @ median (Q=0) with deadband
        _, _, cr_b = crossing_rates_logits(z_bw, Q=0.0, eps=eps_deadband)
        _, _, cr_a = crossing_rates_logits(z_aw, Q=0.0, eps=eps_deadband)
        signflips_logit_delta = float(cr_a - cr_b)
        asym_b = median_crossing_asym(z_bw, Q=0.0, eps=eps_deadband)
        asym_a = median_crossing_asym(z_aw, Q=0.0, eps=eps_deadband)
        med_cross_asym_fisher_delta = fisher_delta(asym_a, asym_b)

        # Residence-time logratio
        res_time_logratio = float(
            mean_log_res_time(z_aw, Q=0.0, eps=eps_deadband)
            - mean_log_res_time(z_bw, Q=0.0, eps=eps_deadband)
        )

        # ACF/short-lag on detrended local windows
        r1_b = acf_1d(zd_bw, 1)[0] if zd_bw.size > 1 else 0.0
        r1_a = acf_1d(zd_aw, 1)[0] if zd_aw.size > 1 else 0.0
        acf1_local_delta = float(r1_a - r1_b)
        shortlag_local_delta = _shortlag_L1(zd_aw, acf_K) - _shortlag_L1(zd_bw, acf_K)

        # Spectral centroid delta on detrended local windows
        fa, Pa = _psd_rfft(zd_aw)
        fb, Pb = _psd_rfft(zd_bw)
        spec_centroid_local_delta = _spectral_centroid(fa, Pa) - _spectral_centroid(
            fb, Pb
        )

        # Local W1 on standardized (scaled by local IQR_b)
        iqr_b = (Qb[-2] - Qb[1]) if len(Qb) >= 4 else (Q90_b - Q10_b)
        w1_scale = max(iqr_b, EPS)
        w1_scaled = _wasserstein_quant(z_aw, z_bw, quantile_coarse_grid, w1_scale)

        # Local RMS logratio on detrended
        rms_b = float(np.sqrt(np.mean(zd_bw**2))) if zd_bw.size else 0.0
        rms_a = float(np.sqrt(np.mean(zd_aw**2))) if zd_aw.size else 0.0
        rms_logratio = np.log(rms_a + EPS) - np.log(rms_b + EPS)

        # Cliff's delta (standardized windows)
        cd = _cliffs_delta(z_aw, z_bw)
        cliffs_fisher_delta = fisher_delta(cd, 0.0)

        # Chow F on clipped local windows
        chow_F = _chow_F(zc_bw, zc_aw)

        # Signed CUSUM on detrended local windows
        cusum_signed = _cusum_signed_stat(zd_bw, zd_aw)

        # Gauss GLR
        gauss_glr = _gaussian_glr(z_bw, z_aw)

        # Diff median jump
        local_diff_median_jump = float(np.median(dz_aw) - np.median(dz_bw))

        # Diff2 MAD logratio
        mad_b2 = 1.4826 * _mad(dd_bw) + EPS
        mad_a2 = 1.4826 * _mad(dd_aw) + EPS
        local_dd_MAD_logratio = float(np.log(mad_a2) - np.log(mad_b2))

        # Diff2 ACF1 delta on detrended local windows
        r1_b2 = acf_1d(ddmw, 1)[0] if ddmw.size > 1 else 0.0
        r1_a2 = acf_1d(ddaw, 1)[0] if ddaw.size > 1 else 0.0
        local_dd_acf1_delta = float(r1_a2 - r1_b2)

        # Diff2 Local W1 scaled by local IQR_before (use coarse grid)
        Qb2 = np.quantile(dd_bw, quantile_coarse_grid)
        idr_local_b = float(Qb2[-1] - Qb2[0]) + EPS
        local_dd_w1_scaled = _wasserstein_quant(
            dd_aw, dd_bw, quantile_coarse_grid, idr_local_b
        )

        return pd.Series(
            {
                "local_median_jump": local_median_jump,
                "local_MAD_logratio": local_scale_logratio,
                "local_slope_jump": slope_jump,
                "local_IDR_logratio": IDR_logratio,
                "local_abs_excd_q95_logit_delta": a1 - b1,
                "local_abs_excd_3_logit_delta": a2 - b2,
                "local_signflips_logit_delta": signflips_logit_delta,
                "local_med_cross_asym_fisher_delta": med_cross_asym_fisher_delta,
                "local_res_time_logratio": res_time_logratio,
                "local_acf1_delta": acf1_local_delta,
                "local_shortlag_L1_delta": shortlag_local_delta,
                "local_spec_centroid_delta": spec_centroid_local_delta,
                "local_w1_scaled": w1_scaled,
                "local_rms_logratio": rms_logratio,
                "local_cliffs_fisher_delta": cliffs_fisher_delta,
                "local_chow_F": chow_F,
                "local_cusum_signed": cusum_signed,
                "local_gauss_glr": gauss_glr,
                "local_diff_median_jump": local_diff_median_jump,
                "local_dd_MAD_logratio": local_dd_MAD_logratio,
                "local_dd_acf1_delta": local_dd_acf1_delta,
                "local_dd_w1_scaled": local_dd_w1_scaled,
            },
            dtype=np.float32,
        )

    # compute per-window, suffix columns, then concat horizontally
    pieces = []
    for w in w_list:
        out_w = g.apply(lambda df: _one_id(df, w)).astype(np.float32)
        out_w = out_w.add_suffix(f"_w{w}")  # suffix all columns for clarity
        pieces.append(out_w)

    out = pd.concat(pieces, axis=1)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# BOUNDARY EDGE BLOCK
# ─────────────────────────────────────────────────────────────────────


def compute_boundary_edge_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    w_edge: int = BOUND_EDGE,  # edge window length
    offsets: tuple[int, ...] = BOUND_OFFSETS,  # after-side offsets to test
) -> pd.DataFrame:
    """
    Compute edge-Window median jump and rank-logit delta at multiple after-side offsets.
    For each id:
      • boundary_dz_o{off}           : median(after[off:off+w_edge]) − median(before[-w_edge:])
      • ranklogit_delta_o{off}       : logit(rank(v_a vs BEFORE)) − logit(rank(v_b vs BEFORE))
      • edge_dz_max_abs, edge_ranklogit_max_abs

    Notes:
      - Uses standardized z (robust to scale/shift).
      - Independent of w_local; only relies on w_edge and offsets.
      - If a window is empty, uses 0.0 for that value.
    """
    prefix = "boundary_edge"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # pull standardized segments as numpy
    z_b = _group_series(X_prep, "standardized", 0).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    z_a = _group_series(X_prep, "standardized", 1).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    ids = z_b.index

    # prebuild column names
    dz_cols = [f"edge_dz_o{off}" for off in offsets]
    rk_cols = [f"edge_ranklogit_delta_o{off}" for off in offsets]

    rows = []
    for i in ids:
        zb = z_b.loc[i]
        za = z_a.loc[i]

        # BEFORE edge median (one-time)
        wb = int(min(w_edge, zb.size))
        z_be = zb[-wb:] if wb > 0 else zb[:0]
        v_b = float(np.median(z_be)) if z_be.size else 0.0

        # Jeffreys-smoothed logit rank of v_b vs BEFORE ECDF (one-time)
        zb_sorted = np.sort(zb) if zb.size else np.array([], dtype=np.float32)
        logit_vb = _ecdf_logit_against_before(v_b, zb_sorted) if zb_sorted.size else 0.0

        # per-offset computations
        dz_vals = []
        rk_vals = []

        for off in offsets:
            # AFTER edge median at offset
            start = int(min(max(off, 0), max(za.size - 1, 0)))
            wa = int(min(w_edge, max(za.size - start, 0)))
            z_ae = za[start : start + wa] if wa > 0 else za[:0]
            v_a = float(np.median(z_ae)) if z_ae.size else 0.0

            # signed jump
            dz = v_a - v_b
            dz_vals.append(dz)

            # rank-logit delta vs BEFORE ECDF
            logit_va = _ecdf_logit_against_before(v_a, zb_sorted)
            rk_vals.append(float(logit_va - logit_vb))

        dz_arr = np.asarray(dz_vals, dtype=np.float32)
        rk_arr = np.asarray(rk_vals, dtype=np.float32)

        # summaries (max by absolute magnitude, keep signed value)
        dz_arg = int(np.argmax(np.abs(dz_arr)))
        rk_arg = int(np.argmax(np.abs(rk_arr)))

        row = {
            **{c: v for c, v in zip(dz_cols, dz_arr.tolist())},
            **{c: v for c, v in zip(rk_cols, rk_arr.tolist())},
            "edge_dz_max_abs": float(dz_arr[dz_arg]),
            "edge_ranklogit_max_abs": float(rk_arr[rk_arg]),
        }
        rows.append(row)

    out = pd.DataFrame(rows, index=ids, dtype=np.float32)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# CURVATURE (SECOND DIFFERENCES) BLOCK
# ─────────────────────────────────────────────────────────────────────


def compute_curvature_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    acf_max_lag: int = ACF_MAX_LAG,
    lbq_m: int = LBQ_M,
) -> pd.DataFrame:
    """
    Global features on second differences (curvature) built from:
      - diff2_standardized (dd) and diff2_detrended (ddm)
      - diff_standardized (d) for curv_vs_slope ratio
    Outputs per id:
      curv_energy_logratio,
      curv_vs_slope_logratio, dd_posrate_logit_delta,
      dd_acf1_delta, dd_shortlag_L1_delta, dd_lbq_z_delta,
      dd_spec_centroid_delta
    """
    prefix = "curvature"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    cols = ["period", "diff2_standardized", "diff2_detrended", "diff_standardized"]
    g = X_prep[cols].groupby(level="id", sort=False)

    def _lbq_z(x: np.ndarray, m: int) -> float:
        m_eff = int(min(m, max(1, x.size - 1)))
        r = acf_1d(x.astype(np.float32), m_eff)
        n = x.size
        ks = np.arange(1, m_eff + 1, dtype=np.float32)
        Q = n * (n + 2.0) * np.sum((r**2) / (n - ks))
        return float((Q - m_eff) / np.sqrt(2.0 * m_eff)) if np.isfinite(Q) else 0.0

    def _one_id(df: pd.DataFrame) -> pd.Series:
        b = df[df["period"] == 0]
        a = df[df["period"] == 1]

        dd_b = b["diff2_standardized"].to_numpy(np.float32, copy=False)
        dd_a = a["diff2_standardized"].to_numpy(np.float32, copy=False)
        ddm_b = b["diff2_detrended"].to_numpy(np.float32, copy=False)
        ddm_a = a["diff2_detrended"].to_numpy(np.float32, copy=False)
        d_b = b["diff_standardized"].to_numpy(np.float32, copy=False)
        d_a = a["diff_standardized"].to_numpy(np.float32, copy=False)

        # Energies
        e_b = float(np.mean(dd_b * dd_b)) if dd_b.size else 0.0
        e_a = float(np.mean(dd_a * dd_a)) if dd_a.size else 0.0
        curv_energy_logratio = float(np.log(e_a + EPS) - np.log(e_b + EPS))

        s_b = float(np.mean(d_b * d_b)) if d_b.size else 0.0
        s_a = float(np.mean(d_a * d_a)) if d_a.size else 0.0
        curv_vs_slope_logratio = float(
            np.log((e_a / (s_a + EPS)) + EPS) - np.log((e_b / (s_b + EPS)) + EPS)
        )

        # Sign dynamics (P(dd > 0) Jeffreys-logit delta)
        pos_b = int(np.sum(dd_b > 0.0))
        m_b = dd_b.size if dd_b.size else 1
        pos_a = int(np.sum(dd_a > 0.0))
        m_a = dd_a.size if dd_a.size else 1
        dd_posrate_logit_delta = jeffreys_logit(pos_a, m_a) - jeffreys_logit(pos_b, m_b)

        # Dependence on detrended curvature
        dd_acf1_delta = float(
            (acf_1d(ddm_a, 1)[0] if ddm_a.size > 1 else 0.0)
            - (acf_1d(ddm_b, 1)[0] if ddm_b.size > 1 else 0.0)
        )
        dd_shortlag_L1_delta = _shortlag_L1(ddm_a, acf_max_lag) - _shortlag_L1(
            ddm_b, acf_max_lag
        )

        m_id = int(min(lbq_m, max(1, ddm_b.size - 1), max(1, ddm_a.size - 1)))
        dd_lbq_z_delta = _lbq_z(ddm_a, m_id) - _lbq_z(ddm_b, m_id)

        # Spectral centroid on ddm
        fa, Pa = _psd_rfft(ddm_a)
        fb, Pb = _psd_rfft(ddm_b)
        dd_spec_centroid_delta = _spectral_centroid(fa, Pa) - _spectral_centroid(fb, Pb)

        return pd.Series(
            {
                "curv_energy_logratio": curv_energy_logratio,
                "curv_vs_slope_logratio": curv_vs_slope_logratio,
                "dd_posrate_logit_delta": dd_posrate_logit_delta,
                "dd_acf1_delta": dd_acf1_delta,
                "dd_shortlag_L1_delta": dd_shortlag_L1_delta,
                "dd_lbq_z_delta": dd_lbq_z_delta,
                "dd_spec_centroid_delta": dd_spec_centroid_delta,
            },
            dtype=np.float32,
        )

    out = g.apply(_one_id).astype(np.float32)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# ROLLING BLOCK
# ─────────────────────────────────────────────────────────────────────


def _cumsums_z(z):
    S = np.empty(z.size + 1, dtype=np.float32)
    S[0] = 0.0
    np.cumsum(z, out=S[1:])
    S2 = np.empty(z.size + 1, dtype=np.float32)
    S2[0] = 0.0
    np.cumsum(z * z, out=S2[1:])
    return S, S2


def _cumsums_dm2(dm):
    Sd = np.empty(dm.size + 1, dtype=np.float32)
    Sd[0] = 0.0
    np.cumsum(dm * dm, out=Sd[1:])
    return Sd


def _cross_prefix(z, eps):
    c = ((z[:-1] <= -eps) & (z[1:] >= eps)) | ((z[:-1] >= eps) & (z[1:] <= -eps))
    C = np.empty(c.size + 1, dtype=np.int32)
    C[0] = 0
    np.cumsum(c.astype(np.int32), out=C[1:])
    return C


# --- small helpers (fast top-k mean on |x|; no full sort) ---------------------
def _topk_mean_abs(x: np.ndarray, k: int) -> float:
    n = x.size
    if n == 0:
        return 0.0
    kk = min(k, n)
    ax = np.abs(x)
    # take k largest by abs via partition (O(n))
    # np.partition keeps the k largest in the last kk positions (unordered)
    idx = np.argpartition(ax, n - kk)[-kk:]
    return float(ax[idx].mean())


# --- rolling jumps from precomputed prefixes ----------------------------------
def _roll_logstd_jump_stats_from_cumsums(
    S: np.ndarray, S2: np.ndarray, w: int, min_pos: int, topk: int
) -> tuple[float, float, float]:
    """Returns (maxpos, minneg, topkabs_mean) for Δ log-std between adjacent windows of size w."""
    n = S.size - 1  # since S is cumsum with S[0]=0 of length n+1
    pos = n - 2 * w + 1
    if n < 2 * w or pos < min_pos:
        return 0.0, 0.0, 0.0
    # rolling mean and var (population) for all windows
    # for t in [0..n-w]: sum = S[t+w]-S[t], mean = sum/w
    sum_w = S[w:] - S[:-w]  # length n-w+1
    mean_w = sum_w / float(w)
    sumsq_w = S2[w:] - S2[:-w]  # length n-w+1
    var_w = np.maximum(sumsq_w / float(w) - mean_w * mean_w, 0.0)  # numerical safety
    logstd_w = np.log(np.sqrt(var_w) + EPS)  # length n-w+1

    # adjacent jump J[t] = R[t] - L[t] with R = logstd[t+w], L = logstd[t], t in [0..n-2w]
    L = logstd_w[:-w]
    R = logstd_w[w:]
    J = R - L
    if J.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(J)), float(np.min(J)), _topk_mean_abs(J, topk)


def _roll_logrms_jump_stats_from_cumsums(
    Sd: np.ndarray, w: int, min_pos: int, topk: int
) -> tuple[float, float, float]:
    """Returns (maxpos, minneg, topkabs_mean) for Δ log-RMS of diff_detrended between adjacent windows of size w."""
    n = Sd.size - 1  # Sd is cumsum of dm^2 with Sd[0]=0
    pos = n - 2 * w + 1
    if n < 2 * w or pos < min_pos:
        return 0.0, 0.0, 0.0
    sumsq_w = Sd[w:] - Sd[:-w]  # length n-w+1
    rms_w = np.sqrt(np.maximum(sumsq_w / float(w), 0.0))
    logrms_w = np.log(rms_w + EPS)

    L = logrms_w[:-w]
    R = logrms_w[w:]
    J = R - L
    if J.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(J)), float(np.min(J)), _topk_mean_abs(J, topk)


def _roll_crossrate_logit_jump_from_prefix(
    C: np.ndarray, w: int, min_pos: int, topk: int
) -> tuple[float, float, float]:
    """
    C is prefix sum over 'cross' of length n-1 (C[0]=0, C[t]=sum_{u< t} cross[u]).
    Window [i, i+w-1] has (w-1) transitions: count = C[i+w-1]-C[i].
    p_hat = (count + 0.5) / ((w-1) + 1.0)   (Jeffreys), logit(p_hat) then adjacent jump.
    """
    # C length = n_trans+1 where n_trans = n-1; valid start i: 0..(n-w)
    n_trans = C.size - 1  # = (n-1)
    n = n_trans + 1
    pos = n - 2 * w + 1
    if n < 2 * w or pos < min_pos or w < 2:
        return 0.0, 0.0, 0.0

    # for i in [0..n-w], transitions idx span [i .. i+w-2] ⇒ count = C[i+w-1]-C[i]
    cnt_w = C[w - 1 :] - C[: -w + 1]  # length n-w+1
    denom = (w - 1) + 1.0  # Jeffreys (w-1 transitions + 1.0)
    p = (cnt_w + 0.5) / denom
    p = np.clip(p, EPS, 1.0 - EPS)
    logit = np.log(p) - np.log(1.0 - p)

    L = logit[:-w]
    R = logit[w:]
    J = R - L
    if J.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(J)), float(np.min(J)), _topk_mean_abs(J, topk)


def _ewvar_last(x, hl):
    """
    EWMA of x^2 with half-life hl; return last value.
    alpha = 1 - 2^(-1/hl)
    """
    if x.size == 0:
        return 0.0
    alpha = 1.0 - 2.0 ** (-1.0 / max(hl, 1.0))
    v = float(x[0] * x[0])
    for xi in x[1:]:
        v = (1 - alpha) * v + alpha * float(xi * xi)
    return v


def compute_rolling_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    windows: tuple = ROLL_WINDOWS,
    min_positions_per_half: int = ROLL_MIN_POS_PER_HALF,
    ewvar_half_lives: tuple = EWVAR_HALFLIVES,
    crossing_rate_deadband: float = CROSSING_RATE_DEADBAND,
    topk: int = ROLL_TOPK,  # <— add this
) -> pd.DataFrame:
    """
    Fast rolling / localized-change features.
      • Rolling log-std jumps on 'clipped' (winsorized standardized)
      • Rolling log-RMS jumps on 'diff_detrended'
      • Rolling crossing-rate (Jeffreys-logit) jumps on 'clipped'
      • EWVAR logratios at the boundary (two half-lives)
    All per-id; deltas = AFTER − BEFORE. Returns float32 DataFrame.
    """
    prefix = "rolling"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    zc_b = _group_series(X_prep, "clipped", 0).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    zc_a = _group_series(X_prep, "clipped", 1).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    dm_b = _group_series(X_prep, "diff_detrended", 0).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    dm_a = _group_series(X_prep, "diff_detrended", 1).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    ids = zc_b.index

    rows = []
    for i in ids:
        zb, za = zc_b.loc[i], zc_a.loc[i]
        dmb, dma = dm_b.loc[i], dm_a.loc[i]
        nb, na = zb.size, za.size

        feats = {}

        # precompute prefixes once per half
        S_b, S2_b = _cumsums_z(zb)
        S_a, S2_a = _cumsums_z(za)
        Sd_b = _cumsums_dm2(dmb)
        Sd_a = _cumsums_dm2(dma)
        C_b = _cross_prefix(zb, crossing_rate_deadband)
        C_a = _cross_prefix(za, crossing_rate_deadband)

        for w in windows:
            # initialize columns for every id (avoid NaNs when invalid)
            feats[f"roll_logstd_jump_w{w}_maxpos_delta"] = np.float32(0.0)
            feats[f"roll_logstd_jump_w{w}_maxneg_delta"] = np.float32(0.0)
            feats[f"roll_logstd_jump_w{w}_topkabs_mean_delta"] = np.float32(0.0)

            feats[f"roll_rms_jump_w{w}_maxpos_delta"] = np.float32(0.0)
            feats[f"roll_rms_jump_w{w}_maxneg_delta"] = np.float32(0.0)
            feats[f"roll_rms_jump_w{w}_topkabs_mean_delta"] = np.float32(0.0)

            feats[f"roll_crossrate_logit_w{w}_maxpos_delta"] = np.float32(0.0)
            feats[f"roll_crossrate_logit_w{w}_maxneg_delta"] = np.float32(0.0)
            feats[f"roll_crossrate_logit_w{w}_topkabs_mean_delta"] = np.float32(0.0)

            # quick feasibility check for both halves
            if (nb - 2 * w + 1) < min_positions_per_half or (
                na - 2 * w + 1
            ) < min_positions_per_half:
                continue

            # log-std (zc)
            smax_b, smin_b, stop_b = _roll_logstd_jump_stats_from_cumsums(
                S_b, S2_b, w, min_positions_per_half, topk
            )
            smax_a, smin_a, stop_a = _roll_logstd_jump_stats_from_cumsums(
                S_a, S2_a, w, min_positions_per_half, topk
            )
            feats[f"roll_logstd_jump_w{w}_maxpos_delta"] = np.float32(smax_a - smax_b)
            feats[f"roll_logstd_jump_w{w}_maxneg_delta"] = np.float32(smin_a - smin_b)
            feats[f"roll_logstd_jump_w{w}_topkabs_mean_delta"] = np.float32(
                stop_a - stop_b
            )

            # log-RMS (diff_detrended)
            rmax_b, rmin_b, rtop_b = _roll_logrms_jump_stats_from_cumsums(
                Sd_b, w, min_positions_per_half, topk
            )
            rmax_a, rmin_a, rtop_a = _roll_logrms_jump_stats_from_cumsums(
                Sd_a, w, min_positions_per_half, topk
            )
            feats[f"roll_rms_jump_w{w}_maxpos_delta"] = np.float32(rmax_a - rmax_b)
            feats[f"roll_rms_jump_w{w}_maxneg_delta"] = np.float32(rmin_a - rmin_b)
            feats[f"roll_rms_jump_w{w}_topkabs_mean_delta"] = np.float32(
                rtop_a - rtop_b
            )

            # crossing-rate (zc, Jeffreys-logit)
            cmax_b, cmin_b, ctop_b = _roll_crossrate_logit_jump_from_prefix(
                C_b, w, min_positions_per_half, topk
            )
            cmax_a, cmin_a, ctop_a = _roll_crossrate_logit_jump_from_prefix(
                C_a, w, min_positions_per_half, topk
            )
            feats[f"roll_crossrate_logit_w{w}_maxpos_delta"] = np.float32(
                cmax_a - cmax_b
            )
            feats[f"roll_crossrate_logit_w{w}_maxneg_delta"] = np.float32(
                cmin_a - cmin_b
            )
            feats[f"roll_crossrate_logit_w{w}_topkabs_mean_delta"] = np.float32(
                ctop_a - ctop_b
            )

        # EWVAR boundary logratios (unchanged)
        for hl in ewvar_half_lives:
            vb = _ewvar_last(zb, hl)
            va = _ewvar_last(za, hl)
            feats[f"ewvar_hl{hl}_logratio"] = np.float32(
                np.log(va + EPS) - np.log(vb + EPS)
            )

        rows.append((i, feats))

    out = pd.DataFrame({idx: f for idx, f in rows}).T
    out.index = ids
    out = out.astype(np.float32)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# AR BLOCK
# ─────────────────────────────────────────────────────────────────────


def _design(y: np.ndarray, p: int):
    n = y.size
    if n <= p:
        return None, None
    Y = y[p:]
    X = np.column_stack(
        [np.ones(n - p, dtype=np.float32)] + [y[p - j : n - j] for j in range(1, p + 1)]
    ).astype(np.float32, copy=False)
    return X, Y


def _ridge_fit(y: np.ndarray, p: int, lam: float):
    X, Y = _design(y, p)
    if X is None:
        return None
    # (X^T X + λ*I_p)^{-1} X^T Y, but do NOT penalize intercept
    XtX = X.T @ X
    reg = np.eye(p + 1, dtype=np.float32) * lam
    reg[0, 0] = 0.0  # intercept unpenalized
    beta = np.linalg.solve(XtX + reg, X.T @ Y).astype(np.float32)
    c = float(beta[0])
    phi = beta[1:].astype(np.float32)
    # train residual variance
    mu = X @ beta
    e = (Y - mu).astype(np.float32)
    sigma2 = float(np.mean(e * e) + EPS)
    return phi, c, sigma2


def _mean_nll_window(y: np.ndarray, p: int, phi: np.ndarray, c: float, sigma2: float):
    X, Y = _design(y, p)
    if X is None:
        return 0.0, np.zeros(0, dtype=np.float32)
    mu = c + X[:, 1:] @ phi
    inv_s2 = 1.0 / (sigma2 + 1e-12)
    resid = (Y - mu).astype(np.float32)
    nll = 0.5 * (np.log(2.0 * np.pi * (sigma2 + 1e-12)) + (resid * resid) * inv_s2)
    return float(np.mean(nll)), resid


def compute_ar_block(
    X_prep: pd.DataFrame,
    force: bool = False,
    inference: bool = False,
    p: int = AR_ORDER,  # AR order
    ridge_lambda: float = AR_RIDGE_LAMBDA,
    score_cap: int = AR_SCORE_CAP,  # equal-length scoring cap H
) -> pd.DataFrame:
    """
    Ridge AR(p) on 'detrended', emit ONLY:
        ar_ridge_nll_logratio = log( meanNLL(AFTER_head) / meanNLL(BEFORE_hold) )

    Train on BEFORE (excluding a length-H holdout at the end), score on:
      • BEFORE_hold = last H of BEFORE
      • AFTER_head  = first H of AFTER
    with H = min(score_cap, floor(nb/2), na).

    No guards: if series are too short, this will raise (as requested).
    """
    prefix = "ar"
    cache = _latest_cache(prefix)
    if cache and not force and not inference:
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    EPS = globals().get("EPS", 1e-8)

    # pull detrended series
    zb = _group_series(X_prep, "detrended", 0).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    za = _group_series(X_prep, "detrended", 1).apply(
        lambda s: s.to_numpy(np.float32, copy=False)
    )
    ids = zb.index

    rows = []
    for i in ids:
        b = zb.loc[i]
        a = za.loc[i]
        nb, na = b.size, a.size

        # equal-length scoring window
        H = int(min(score_cap, nb // 2, na))

        # split BEFORE into train + hold (no guards)
        n_train_b = nb - H
        b_train = b[:n_train_b]
        b_hold = b[nb - H :]

        # fit on BEFORE-train (no guards)
        phi_b, c_b, s2_b = _ridge_fit(b_train, p, ridge_lambda)

        # mean NLL on BEFORE_hold and AFTER_head
        nll_bh, _ = _mean_nll_window(b_hold, p, phi_b, c_b, s2_b)
        nll_af, _ = _mean_nll_window(a[:H], p, phi_b, c_b, s2_b)

        nll_logratio = float(np.log((nll_af + EPS) / (nll_bh + EPS)))

        rows.append((i, {"ar_ridge_nll_logratio": np.float32(nll_logratio)}))

    out = pd.DataFrame({idx: feats for idx, feats in rows}).T
    out.index = ids
    out = out.astype(np.float32)

    if not inference:
        _save_cache(out, prefix)
    return out


# ─────────────────────────────────────────────────────────────────────
# Build features wrapper
# ─────────────────────────────────────────────────────────────────────

FEATURE_BLOCKS = {
    "moments": compute_moments_block,
    "quantiles": compute_quantiles_block,
    "rates": compute_rates_block,
    "autocorrelation": compute_autocorr_block,
    "tests_distances": compute_tests_distances_block,
    "frequency": compute_frequency_block,
    "differences": compute_differences_block,
    "absolute": compute_absolute_block,
    "squared": compute_squared_block,
    "boundary_local": compute_boundary_local_block,
    "boundary_edge": compute_boundary_edge_block,
    "curvature": compute_curvature_block,
    "rolling": compute_rolling_block,
    "ar": compute_ar_block,
}


def build_features(
    X_train: pd.DataFrame,
    force_prep: bool = False,
    force_all: bool = False,
    force: dict[str, bool] = None,
    inference: bool = False,
    feature_blocks: dict = FEATURE_BLOCKS,
    feat_cache_dir: str = FEAT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Build and return a wide per-id feature table by:
      1) Preprocessing the raw, long-format input (delegated to `build_preprocessed`).
      2) Computing each registered feature block in `BLOCKS` (each block caches itself).
      3) Merging all block outputs on the index (id).

    Parameters
    ----------
    X : pd.DataFrame
        Raw input in the expected MultiIndex (id, time) format for preprocessing.
    force_prep : bool, optional
        If True, recompute preprocessing and ignore the upstream preprocess cache.
    force_all : bool, optional
        If True, recompute *all* feature blocks (overrides per-block cache).
    force : dict[str, bool] | None, optional
        Per-block override, e.g. {'moments': True, 'quantiles': False}.
        Only used when `force_all` is False. Keys must match `BLOCKS` names.

    Returns
    -------
    pd.DataFrame
        Wide feature matrix with one row per id and columns from all blocks.
    """
    # Create the feature folder
    feat_cache_dir = Path(feat_cache_dir)
    feat_cache_dir.mkdir(parents=True, exist_ok=True)

    # Decide which blocks to recompute. If force_all is True, force all blocks.
    force = force or {}
    if inference:
        force_prep = True
        force_all = True
    if force_all:
        force = {name: True for name in feature_blocks}

    # Do not recompute anything if everything in force is False
    prefix = "all"
    cache = _latest_cache(prefix)
    if cache and not force_prep and not any(force.values()):
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    # 1) Preprocess raw X
    X_prep = build_preprocessed(X_train, force_prep, inference)

    # Sanity check 1
    detect_non_finite(X_prep)

    # 2) Compute each block (respect per-block force flags)
    parts = []
    for name, fn in feature_blocks.items():
        part = fn(X_prep, force.get(name, False), inference)
        parts.append(part)

    # 3) Merge all blocks on id (inner join ensures only ids present in all blocks remain)
    feats = parts[0].join(parts[1:], how="inner")

    # Sanity check 2
    ids = X_train.index.get_level_values("id").unique().size
    n = feats.index.size
    if ids != n:
        raise ValueError(f"Feature table has {n} ids, but input had {ids} ids")
    detect_non_finite(feats)

    # Save features
    if not inference:
        _save_cache(feats, prefix)

    return feats
