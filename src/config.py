from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────

RANDOM_STATE = 69
np.random.seed(RANDOM_STATE)

EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────
# INFER CONFIG
# ─────────────────────────────────────────────────────────────────────

INFERENCE_MODE = "full"

# ─────────────────────────────────────────────────────────────────────
# META LEARNER CONFIG
# ─────────────────────────────────────────────────────────────────────

BASE_LEARNERS = (
    "xgb_main",
    "xgb_lite",
    "lgb_main",
    "cat_main",
)

N_SEEDS = 3
TOP_SEEDS = 2
FULL_REFIT = True
FULL_HP_SELECTION = "consensus"


# ─────────────────────────────────────────────────────────────────────
# TRAIN CONFIG
# ─────────────────────────────────────────────────────────────────────


N_OPTUNA_TRIALS = 5
K_OUTER = 5
K_MAX_INNER = 5
K_STOP_INNER = 1

TOPK_MIN_AUC = 0.52
TOPK_FEATURES = 60
TOPK_ALWAYS_KEEP = []

EXCLUDE_FEATURE_KEYWORDS = ["logit", "fisher", "logratio", "scaled"]


MAX_BIN = 64
EARLY_STOPPING = 100

MODEL_DIR = Path("resources/model")


# ─────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION CONFIG
# ─────────────────────────────────────────────────────────────────────


# Moments
QUANTILE_COARSE_GRID = [0.10, 0.25, 0.50, 0.75, 0.90]

# Quantiles
QUANTILE_FINE_GRID = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
TOP_K = 5

# Crossing rates
W_FANO = 50
CROSSING_RATE_DEADBAND = 0.1

# Autocorrelation
ACF_MAX_LAG = 10
LBQ_M = 20

# Tests & Distances
JS_QUANTILE_BINS = np.linspace(0.0, 1.0, 33)
MMD_MAX_N = 512

# Frequency
FREQ_BANDS = ((0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50))
DWT_WAVELET = "db2"
DWT_LEVEL = 3
ENTROPY_M1, ENTROPY_M2 = 3, 5
ENTROPY_TAU = 1

# Boundary
BOUND_EDGE = 5
BOUND_WINDOW_SIZES = [32, 128]
BOUND_SKIP_AFTER = 0
BOUND_ACF_MAX_LAG = 6
ARCH_L = 5
BOUND_OFFSETS = (0, 8, 16, 32)

# Rolling
ROLL_WINDOWS = (10, 20, 50, 100, 200)
ROLL_MIN_POS_PER_HALF = 20
ROLL_TOPK = 3
EWVAR_HALFLIVES = (200, 400)

# AR
AR_ORDER = 1
AR_RIDGE_LAMBDA = 1.0
AR_SCORE_CAP = 256


# ─────────────────────────────────────────────────────────────────────
# PREPROCESS CONFIG
# ─────────────────────────────────────────────────────────────────────


# Floors computed as the 5% bottom quantile of s0, s_dz, s_abs, s_dd
S0_FLOOR = 0.0008216
S_DZ_FLOOR = 1.0909
S_ABS_FLOOR = 0.5555
S_DD_FLOOR = 1.6383
CLIP_QLOW, CLIP_QHIGH = 0.002, 0.998
CLIP_MIN_WIDTH = 1.0
CLIP_DEFAULT_BAND = 7.0  # computed as the 0.2% and 99.8% quantiles of z_before
FEAT_CACHE_DIR = Path("resources/features")
