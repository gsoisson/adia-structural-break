# ─────────────────────────────────────────────────────────────────────
#  0. imports
# ─────────────────────────────────────────────────────────────────────

# Standard library
import sys
import types
import json
from pathlib import Path
from joblib import dump, load
from typing import List, Tuple
import hashlib


# Data handling
import numpy as np
import pandas as pd

# Machine learning frameworks
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from xgboost.callback import EarlyStopping as XGBEarlyStop
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, BaseCrossValidator
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

# Hyperparameter optimization
import optuna
from optuna import Trial


# Config
from src.config import (
    MAX_BIN,
    EARLY_STOPPING,
    K_OUTER,
    K_MAX_INNER,
    K_STOP_INNER,
    N_OPTUNA_TRIALS,
    TOPK_FEATURES,
    TOPK_MIN_AUC,
    TOPK_ALWAYS_KEEP,
    EXCLUDE_FEATURE_KEYWORDS,
    MODEL_DIR,
    RANDOM_STATE,
    FULL_REFIT,
    FULL_HP_SELECTION,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)

# ─────────────────────────────────────────────────────────────────────
#  1. build estimator
# ─────────────────────────────────────────────────────────────────────


def make_estimator(
    model_type: str,
    params: dict,
) -> ClassifierMixin:
    """
    Return an untrained binary classifier with default settings updated by `params`.
    """
    params = params.copy()  # don't mutate caller's dict

    # ---------------- XGBoost ----------------
    if "xgb" in model_type:
        if xgb is None:
            raise ImportError("XGBoost not installed in the runner.")

        defaults = dict(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            verbosity=0,
            max_bin=MAX_BIN,
            n_jobs=-1,
            use_label_encoder=False,
            tree_method="hist",  # faster histogram optimized
        )
        defaults.update(params)

        return xgb.XGBClassifier(**defaults)

    # ---------------- LightGBM -------------
    elif "lgb" in model_type:
        if lgb is None:
            raise ImportError("LightGBM not installed in the runner.")
        defaults = dict(
            objective="binary",
            metric="auc",
            random_state=RANDOM_STATE,
            verbosity=-1,
            max_bin=MAX_BIN,  # controls histogram granularity
            n_jobs=-1,  # controls how many threads you use, -1 is for max
            force_row_wise=True,  # parallelizes the computation of the histogram along rows instead of columns
        )
        defaults.update(params)

        return lgb.LGBMClassifier(**defaults)

    # ---------------- CatBoost -------------------------------
    elif "cat" in model_type:
        if CatBoostClassifier is None:
            raise ImportError("CatBoost not installed in the runner.")
        defaults = dict(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    else:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────
#  2. build pipeline
# ─────────────────────────────────────────────────────────────────────


def _fast_auc_binary(x: np.ndarray, y: np.ndarray) -> float:
    """Mann–Whitney U / (n_pos*n_neg). NaN-safe (drops NaNs)."""
    mask = np.isfinite(x)
    x = x[mask]
    yb = y[mask]
    n1 = int((yb == 1).sum())
    n0 = int((yb == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(x)  # average ranks for ties
    R1 = r[yb == 1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    return float(U1 / (n1 * n0))


class TopKUnivariateAUC(BaseEstimator, TransformerMixin):
    """
    Keep top-K features by univariate AUC with the label.
    Always keep any columns listed in `always_keep` (if present in X).
    """

    def __init__(
        self,
        k: int = TOPK_FEATURES,
        min_auc: float = TOPK_MIN_AUC,
        always_keep=None,
    ):
        self.k = int(k)
        self.min_auc = float(min_auc)
        self.always_keep = list(always_keep) if always_keep is not None else []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.original_n_features_ = X.shape[1]

        aucs = np.array(
            [
                _fast_auc_binary(X[c].to_numpy(dtype=float), y.to_numpy())
                for c in X.columns
            ]
        )
        score = np.abs(aucs - 0.5)

        order = np.argsort(score)[::-1]
        keep_idx = order[: self.k]
        keep_idx = keep_idx[score[keep_idx] >= (self.min_auc - 0.5)]
        selected = list(X.columns[keep_idx])

        always = [c for c in self.always_keep if c in X.columns]
        selected_set = set(selected) | set(always)
        rest = sorted(
            [c for c in selected_set if c not in always],
            key=lambda c: (-score[X.columns.get_loc(c)], c),
        )
        self.keep_cols_ = always + rest

        # bookkeeping
        self.n_features_selected_ = len(self.keep_cols_)
        self.selection_rate_ = self.n_features_selected_ / max(
            1, self.original_n_features_
        )
        self.feature_scores_ = pd.Series(
            score, index=X.columns
        )  # optional: for inspection
        return self

    def transform(self, X: pd.DataFrame):
        cols = [c for c in self.keep_cols_ if c in X.columns]
        return X.loc[:, cols]

    def get_feature_names_out(self):
        return np.asarray(self.keep_cols_)


# Avoid joblib unpickle errors
_main = sys.modules.get("__main__")
if _main is None:
    _main = types.ModuleType("__main__")
    sys.modules["__main__"] = _main
setattr(_main, "TopKUnivariateAUC", TopKUnivariateAUC)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Wrap a StandardScaler (or any sklearn transformer) so that both fit_transform
    and transform return a DataFrame with the original columns & index.

    Set `exclude` to a list of feature names that should be passed through
    without scaling (if present in X).
    """

    def __init__(self, scaler=None, exclude=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.exclude = set(exclude) if exclude is not None else set()

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)

        # self.exclude is now a list of *keywords*, not exact column names
        kw = [k.lower() for k in self.exclude]

        # columns to exclude = any keyword appears as a substring of the column name
        self.exclude_ = [c for c in self.columns_ if any(k in c.lower() for k in kw)]
        self.scale_cols_ = [c for c in self.columns_ if c not in self.exclude_]

        # fit on the subset to be scaled (if any)
        if len(self.scale_cols_) > 0:
            self.scaler.fit(X[self.scale_cols_], y)

        return self

    def transform(self, X):
        # align columns to training order; missing columns will raise KeyError
        X = X.loc[:, self.columns_]
        if len(self.scale_cols_) > 0:
            Xt_scaled = self.scaler.transform(X[self.scale_cols_])
            X_out = X.copy()
            X_out.loc[:, self.scale_cols_] = Xt_scaled
        else:
            X_out = X.copy()
        return pd.DataFrame(X_out.values, columns=self.columns_, index=X.index)

    def get_feature_names_out(self):
        return np.asarray(self.columns_)


# Avoid joblib unpickle errors
setattr(_main, "DataFrameScaler", DataFrameScaler)


def make_pipeline(model_type: str, params: dict) -> Pipeline:
    """
    Construct a sklearn Pipeline that standardizes features then applies
    the specified model with given params.

    [ TopK Filter ] → [ StandardScaler ] → [ model ]
    """
    # There is one feature selection step I did which isn't shown here:
    # After computing features I computed the correlation for each feature pair
    # and deleted the ones with >0.98 in this file directly.
    topk_auc = TopKUnivariateAUC(
        k=TOPK_FEATURES, min_auc=TOPK_MIN_AUC, always_keep=TOPK_ALWAYS_KEEP
    )
    scaler = DataFrameScaler(exclude=EXCLUDE_FEATURE_KEYWORDS)
    model = make_estimator(model_type, params)
    return Pipeline(
        [
            ("topk_auc", topk_auc),
            ("scaler", scaler),
            ("model", model),
        ]
    )


# ─────────────────────────────────────────────────────────────────────
#  4. one-fold fit util
# ─────────────────────────────────────────────────────────────────────


def fit_one_fold(
    model_type: str,
    pipeline: Pipeline,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xva: pd.DataFrame,
    yva: pd.Series,
    early_stopping: int = EARLY_STOPPING,
) -> Pipeline:
    """
    Fit the pipeline on one CV fold, but transform eval_set through the
    *fitted preprocessors* before passing it to the final estimator.
    Prevents feature-mismatch with steps like TopK selectors.
    """
    # split pipeline into [preprocessors] and [final model]
    pre = Pipeline(pipeline.steps[:-1])  # e.g. ("topk_auc","scaler")
    model = pipeline.steps[-1][1]  # the estimator instance

    # fit preprocessors on training split only (CV-safe) and transform both sets
    Xtr_p = pre.fit_transform(Xtr, ytr)
    Xva_p = pre.transform(Xva)

    # Define model specific kwargs
    fit_kwargs = {}

    if "xgb" in model_type:
        fit_kwargs["eval_set"] = [(Xva_p, yva)]
        (
            model.set_params(
                callbacks=[
                    XGBEarlyStop(rounds=early_stopping, save_best=True, maximize=True)
                ]
            ),
        )
        fit_kwargs["verbose"] = 0  # silence XGBoost logging
    elif "lgb" in model_type:
        fit_kwargs["eval_set"] = [(Xva_p, yva)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping, verbose=False)]

    elif "cat" in model_type:
        fit_kwargs["eval_set"] = (Xva_p, yva)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = early_stopping

    # Fit final model with early stopping on the *transformed* eval set
    model.fit(Xtr_p, ytr, **fit_kwargs)

    # rebuild a single pipeline carrying the fitted preprocessors + model
    fitted = Pipeline(pre.steps + [("model", model)])
    return fitted


# ─────────────────────────────────────────────────────────────────────
#  5. hyper-parameter tuning
# ─────────────────────────────────────────────────────────────────────


def default_param_grid(model_type: str):
    """Optuna search space for each model family."""
    if model_type == "xgb_main":
        return {
            "learning_rate": (0.02, 0.06, True),
            "max_depth": (3, 5),
            "min_child_weight": (70.0, 180.0, True),
            "gamma": (1.0, 3.0, True),
            "subsample": (0.65, 0.90),
            "colsample_bytree": (0.50, 0.70),
            "colsample_bylevel": (0.70, 0.90),
            "colsample_bynode": (0.70, 0.90),
            "reg_alpha": (0.5, 4.0, True),
            "reg_lambda": (8.0, 40.0, True),
            "scale_pos_weight": (0.9, 1.2),
        }

    elif model_type == "xgb_lite":
        return {
            "learning_rate": (0.035, 0.07, True),
            "max_depth": (3, 3),  # not tuned, keep truly shallow
            "min_child_weight": (120.0, 250.0, True),
            "gamma": (1.4, 3.5, True),
            "subsample": (0.60, 0.80),
            "colsample_bytree": (0.45, 0.62),
            "colsample_bylevel": (0.70, 0.90),
            "colsample_bynode": (0.70, 0.90),
            "reg_alpha": (1.0, 4.0, True),
            "reg_lambda": (15.0, 40.0, True),
            "scale_pos_weight": (0.9, 1.2),
        }

    elif model_type == "lgb_main":
        return {
            "learning_rate": (0.02, 0.06, True),
            "max_depth": (4, 6),
            "num_leaves": (24, 48),  # consistent with depth 4–6
            "min_child_samples": (40, 120),
            "min_split_gain": (0.0, 0.20),
            "feature_fraction": (0.60, 0.85),
            "bagging_fraction": (0.60, 0.85),
            "lambda_l1": (0.2, 2.5, True),
            "lambda_l2": (5.0, 25.0, True),
            "scale_pos_weight": (0.9, 1.2),
            # ⬇️ constants (not tuned)
            "extra_trees": True,
            "bagging_freq": 1,
        }

    elif model_type == "cat_main":
        return {
            "learning_rate": (0.02, 0.05, True),
            "depth": (3, 5),
            "l2_leaf_reg": (15.0, 60.0, True),
            "bootstrap_type": ["Bayesian"],  # fix scheme; tune temperature
            "bagging_temperature": (1.2, 2.6),
            "rsm": (0.50, 0.70),
            "random_strength": (1.2, 2.2),
            "min_data_in_leaf": (150, 260),
        }
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")


def optuna_objective(
    trial: Trial,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Index,
    cv: BaseCrossValidator,
) -> tuple[float, dict]:
    """
    Optuna objective for each model family (and variants).
    Supports:
      • XGBoost
      • LightGBM
      • CatBoost
    Returns (mean OOF AUC, sampled_hp_dict).
    """
    grid = default_param_grid(model_type)

    # ----- strict helpers: no fallbacks -----
    def ffloat(name):
        rng = grid[name]
        if isinstance(rng, tuple) and len(rng) == 3:
            return trial.suggest_float(name, rng[0], rng[1], log=rng[2])
        return trial.suggest_float(name, rng[0], rng[1])

    def fint(name):
        rng = grid[name]
        return trial.suggest_int(name, rng[0], rng[1])

    def fcat(name):
        return trial.suggest_categorical(name, grid[name])

    is_xgb = "xgb" in model_type
    is_lgb = "lgb" in model_type
    is_cat = "cat" in model_type

    # ----------------- XGBoost -----------------
    if is_xgb:
        hp = dict(
            learning_rate=ffloat("learning_rate"),
            max_depth=fint("max_depth"),
            min_child_weight=ffloat("min_child_weight"),
            gamma=ffloat("gamma"),
            reg_alpha=ffloat("reg_alpha"),
            reg_lambda=ffloat("reg_lambda"),
            subsample=ffloat("subsample"),
            colsample_bytree=ffloat("colsample_bytree"),
            colsample_bylevel=ffloat("colsample_bylevel"),
            colsample_bynode=ffloat("colsample_bynode"),
            scale_pos_weight=ffloat("scale_pos_weight"),
        )

    # ----------------- LightGBM -----------------
    elif is_lgb:
        hp = dict(
            learning_rate=ffloat("learning_rate"),
            max_depth=fint("max_depth"),
            num_leaves=fint("num_leaves"),
            min_child_samples=fint("min_child_samples"),
            min_split_gain=ffloat("min_split_gain"),
            feature_fraction=ffloat("feature_fraction"),
            bagging_fraction=ffloat("bagging_fraction"),
            lambda_l1=ffloat("lambda_l1"),
            lambda_l2=ffloat("lambda_l2"),
            scale_pos_weight=ffloat("scale_pos_weight"),
            extra_trees=grid["extra_trees"],
            bagging_freq=grid["bagging_freq"],
        )

    # ----------------- CatBoost -----------------
    elif is_cat:
        hp = dict(
            learning_rate=ffloat("learning_rate"),
            depth=fint("depth"),
            l2_leaf_reg=ffloat("l2_leaf_reg"),
            rsm=ffloat("rsm"),
            random_strength=ffloat("random_strength"),
            min_data_in_leaf=fint("min_data_in_leaf"),
            bootstrap_type=fcat("bootstrap_type"),
        )
        if hp["bootstrap_type"] == "Bayesian":
            hp["bagging_temperature"] = ffloat("bagging_temperature")
        elif hp["bootstrap_type"] == "Bernoulli":
            hp["subsample"] = ffloat("subsample")
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")

    # ensure every grid key is represented either as a sampled HP or a constant we injected
    expected = set(grid.keys())
    present = set(hp.keys())
    missing = expected - present
    if missing:
        raise KeyError(f"Optuna grid keys not used for {model_type}: {sorted(missing)}")

    # ----------------- CV training -----------------
    pipe = make_pipeline(model_type, hp)
    oof = np.zeros(len(X_train), dtype=float)

    for tr_idx, va_idx in cv.split(X_train, y_train, groups):
        pipe_fold = fit_one_fold(
            model_type,
            pipe,
            X_train.iloc[tr_idx],
            y_train.iloc[tr_idx],
            X_train.iloc[va_idx],
            y_train.iloc[va_idx],
        )
        oof[va_idx] = pipe_fold.predict_proba(X_train.iloc[va_idx])[:, 1]

    return roc_auc_score(y_train, oof), hp


def default_params(model_type: str):
    """Default parameters if no optuna tuning."""
    if model_type == "xgb_main":
        return {
            "booster": "gbtree",
            "n_estimators": 2400,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 110.0,
            "gamma": 1.8,
            "subsample": 0.75,
            "colsample_bytree": 0.58,
            "colsample_bylevel": 0.80,
            "colsample_bynode": 0.80,
            "reg_alpha": 1.5,
            "reg_lambda": 18.0,
            "scale_pos_weight": 1.0,
        }

    elif model_type == "xgb_lite":
        return {
            "booster": "gbtree",
            "n_estimators": 1800,
            "learning_rate": 0.050,
            "max_depth": 3,
            "min_child_weight": 170.0,
            "gamma": 2.2,
            "subsample": 0.68,
            "colsample_bytree": 0.52,
            "colsample_bylevel": 0.80,
            "colsample_bynode": 0.80,
            "reg_alpha": 2.0,
            "reg_lambda": 25.0,
            "scale_pos_weight": 1.0,
        }

    elif model_type == "lgb_main":
        return {
            "boosting_type": "gbdt",
            "n_estimators": 2400,
            "learning_rate": 0.035,
            "max_depth": 5,
            "num_leaves": 32,
            "min_child_samples": 64,
            "min_sum_hessian_in_leaf": 5.0,
            "min_split_gain": 0.1,
            "feature_fraction": 0.75,
            "bagging_freq": 1,
            "bagging_fraction": 0.75,
            "lambda_l1": 1.0,
            "lambda_l2": 10.0,
            "extra_trees": True,
            "scale_pos_weight": 1.0,
            "force_row_wise": True,
        }

    elif model_type == "cat_main":
        return {
            "n_estimators": 2200,
            "learning_rate": 0.035,
            "depth": 4,
            "l2_leaf_reg": 30.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.8,
            "rsm": 0.58,
            "random_strength": 1.7,
            "min_data_in_leaf": 200,
            "od_type": "Iter",
            "od_wait": 100,
        }
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")


def tune_params(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Index,
    cv: BaseCrossValidator,
    n_optuna_trials: int,
) -> dict:
    """Run Optuna and return best hyper-param dict."""
    if n_optuna_trials == 0 or "dart" in model_type:
        return default_params(model_type)

    def obj(trial):
        score, _ = optuna_objective(trial, model_type, X_train, y_train, groups, cv)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(obj, n_trials=n_optuna_trials, show_progress_bar=False)

    _, best_hp = optuna_objective(
        study.best_trial, model_type, X_train, y_train, groups, cv
    )
    return best_hp


# ─────────────────────────────────────────────────────────────────────
#  6. cross-validation helpers
# ─────────────────────────────────────────────────────────────────────
def _outer_cv(K: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)


def _inner_cv(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    g_tr: pd.Index,
    seed: int,
    fold_idx: int,
    K_max_inner: int,
    K_stop_inner: int,
) -> BaseCrossValidator:
    """
    Yield only the first K_stop_inner folds from a StratifiedGroupKFold(n_splits=K_max_inner).
    - K_max_inner=2, K_stop_inner=1  → single 50/50 holdout
    - K_max_inner=5, K_stop_inner=1  → single ~80/20 holdout
    - K_max_inner=5, K_stop_inner=3  → take the first 3 inner folds
    """
    if not (1 <= K_stop_inner <= K_max_inner):
        raise ValueError("Require 1 <= K_stop_inner <= K_max_inner")

    rng = seed + 69 + fold_idx

    # Fast path: single holdout
    if K_stop_inner == 1:
        sgk = StratifiedGroupKFold(n_splits=K_max_inner, shuffle=True, random_state=rng)
        tr_idx, va_idx = next(sgk.split(X_tr, y_tr, g_tr))

        class _One(BaseCrossValidator):
            def get_n_splits(self, *_args, **_kwargs): return 1
            def split(self, *_args, **_kwargs): yield tr_idx, va_idx

        return _One()

    # General case: lazily yield first K_stop_inner folds
    class _LimitedCV(BaseCrossValidator):
        def __init__(self, n_splits: int, n_take: int, random_state: int):
            self.n_splits, self.n_take, self.random_state = n_splits, n_take, random_state
        def get_n_splits(self, *_args, **_kwargs): return self.n_take
        def split(self, X=None, y=None, groups=None):
            sgk = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            for i, (tr, va) in enumerate(sgk.split(X, y, groups)):
                if i >= self.n_take: break
                yield tr, va

    return _LimitedCV(K_max_inner, K_stop_inner, rng)


# ─────────────────────────────────────────────────────────────────────
#  7. cache: layout & IO
# ─────────────────────────────────────────────────────────────────────

def _fold_dir(root: Path) -> Path:
    d = Path(root) / "fold_models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _fold_paths(root: Path, k: int) -> dict:
    d = _fold_dir(root)
    return {"pipe": d / f"fold_{k}.joblib", "hp": d / f"fold_{k}_hp.json"}

def _save_fold(root: Path, k: int, pipe: Pipeline, hp: dict) -> None:
    p = _fold_paths(root, k)
    dump(pipe, p["pipe"])
    with open(p["hp"], "w") as f:
        json.dump(hp, f)

def _save_base_metrics(
    root: Path,
    fold_metrics: List[dict],   # [{fold, train_auc, val_auc, best_iter?}]
    auc_oof: float,
    auc_train_full: float,
) -> None:
    payload = {
        "fold_metrics": fold_metrics,
        "auc_oof": float(auc_oof),
        "auc_train_full": None if auc_train_full is None else float(auc_train_full),
    }
    (Path(root) / "metrics.json").write_text(json.dumps(payload, indent=2))

def _load_base_metrics(root: Path) -> dict:
    p = Path(root) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────
#  8. cache: fingerprint & validation
# ─────────────────────────────────────────────────────────────────────

def _fingerprint_base(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    seed: int,
    params: dict,
) -> Tuple[dict, str]:
    """Stable signature for the CV cache."""
    sig = {
        "model_type": model_type,
        "seed": int(seed),
        "n_rows": int(len(X)),
        "n_cols": int(X.shape[1]),
        "cols": list(map(str, X.columns)),   # preserve order
        "params": params or {},
        "y_sum": int(np.asarray(y).sum()),
        "y_len": int(len(y)),
    }
    raw = json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sig, hashlib.md5(raw).hexdigest()

def _save_base_cache_meta(root: Path, model_type: str, sig: dict, sig_md5: str) -> None:
    dump({"signature": sig, "signature_md5": sig_md5}, Path(root) / f"cvmeta.joblib")

def _has_full_cv_cache(
    root: Path,
    model_type: str,
    K_outer: int,
    N: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    K_max_inner: int,
    K_stop_inner: int,
    n_optuna_trials: int,
) -> bool:
    root = Path(root)

    # 1) OOF exists and length matches
    oof_p = root / f"oof.npy"
    if not oof_p.exists(): return False
    try:
        oof = np.load(oof_p)
        if oof.shape[0] != N: return False
    except Exception:
        return False

    # 2) Every fold has pipe + hp
    for k in range(K_outer):
        p = _fold_paths(root, k)
        if not (Path(p["pipe"]).exists() and Path(p["hp"]).exists()):
            return False

    # 3) Consolidated metrics
    if not (root / "metrics.json").exists(): return False

    # 4) Signature matches
    meta_p = root / f"cvmeta.joblib"
    if not meta_p.exists(): return False
    cached = load(meta_p)  # assume readable
    _, md5_now = _fingerprint_base(
        X=X_train,
        y=y_train,
        model_type=model_type,
        seed=seed,
        params={
            "K_outer": K_outer,
            "K_max_inner": K_max_inner,
            "K_stop_inner": K_stop_inner,
            "n_optuna_trials": n_optuna_trials,
        },
    )
    return cached.get("signature_md5") == md5_now

def _load_cv_cache(root: Path, model_type: str, K_outer: int):
    """
    Returns:
      oof, fold_hps, val_aucs, best_iters, fold_metrics, auc_oof, auc_train_full
    Assumes cache is correct (already validated).
    """
    root = Path(root)

    # OOF
    oof = np.load(root / f"oof.npy")

    # Fold HPs
    fold_hps = []
    for k in range(K_outer):
        with open(_fold_paths(root, k)["hp"], "r") as f:
            fold_hps.append(json.load(f))

    # Consolidated metrics
    meta = _load_base_metrics(root) or {}
    fold_metrics = meta.get("fold_metrics", [])
    val_aucs = [float(m.get("val_auc", np.nan)) for m in fold_metrics]
    best_iters = [int(m["best_iter"]) for m in fold_metrics if m.get("best_iter") is not None]
    auc_oof = float(meta["auc_oof"]) if "auc_oof" in meta else None
    auc_train_full = float(meta["auc_train_full"]) if "auc_train_full" in meta else None

    return oof, fold_hps, val_aucs, best_iters, fold_metrics, auc_oof, auc_train_full


# ─────────────────────────────────────────────────────────────────────
#  9. model-specific utilities
# ─────────────────────────────────────────────────────────────────────

def _extract_best_iteration(pipe: Pipeline, model_type: str) -> int:
    est = pipe.named_steps.get("model")
    if est is None: return None

    if "xgb" in model_type:
        bi = getattr(est, "best_iteration", None)
        if bi is None:
            booster = getattr(est, "get_booster", lambda: None)()
            bi = getattr(booster, "best_ntree_limit", None)
        return int(bi) if bi else None

    if "lgb" in model_type:
        bi = getattr(est, "best_iteration_", None)
        return int(bi) if bi else None

    if "cat" in model_type:
        bi = est.get_best_iteration()
        return int(bi) if bi else None

    return None


def _tune_and_fit_one_fold(
    model_type: str,
    X_tr: pd.DataFrame, y_tr: pd.Series, g_tr: pd.Index,
    X_va: pd.DataFrame, y_va: pd.Series,
    inner_cv: BaseCrossValidator,
    n_optuna_trials: int,
) -> tuple[Pipeline, dict, float, float, int]:
    hp = tune_params(
        model_type=model_type,
        X_train=X_tr,
        y_train=y_tr,
        groups=g_tr,
        cv=inner_cv,
        n_optuna_trials=n_optuna_trials,
    )
    pipe = make_pipeline(model_type, hp)
    pipe = fit_one_fold(model_type, pipe, X_tr, y_tr, X_va, y_va)

    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_va = pipe.predict_proba(X_va)[:, 1]
    auc_tr = float(roc_auc_score(y_tr, p_tr))
    auc_va = float(roc_auc_score(y_va, p_va))
    best_iter = _extract_best_iteration(pipe, model_type)

    return pipe, hp, auc_tr, auc_va, best_iter


def _select_full_hps(mode: str, model_type: str, fold_hps: List[dict], val_aucs: List[float]) -> dict:
    if mode == "best_outer":
        j = int(np.nanargmax(val_aucs))
        return fold_hps[j]
    if mode == "consensus":
        out = {}
        for k in fold_hps[0].keys():
            vals = [hp[k] for hp in fold_hps]
            if all(isinstance(v, (bool, np.bool_)) for v in vals):
                out[k] = sum(bool(v) for v in vals) >= (len(vals) / 2)
            elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                med = float(np.median(vals))
                out[k] = int(round(med)) if all(isinstance(v, (int, np.integer)) for v in vals) else med
    else:
        out = default_params(model_type)
    return out


# ─────────────────────────────────────────────────────────────────────
#  6. main train_model() entry point
# ─────────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,                     # 'xgb_*', 'lgb_*', 'cat_*'
    K_outer: int = K_OUTER,
    K_max_inner: int = K_MAX_INNER,
    K_stop_inner: int = K_STOP_INNER,
    n_optuna_trials: int = N_OPTUNA_TRIALS,
    seed: int = RANDOM_STATE,
    model_dir: str = MODEL_DIR,
    full_refit: bool = FULL_REFIT,
    full_hp_selection: str = FULL_HP_SELECTION,  # 'default' | 'best_outer' | 'consensus'
) -> np.ndarray:
    """
    If cache complete:
      • Print "Using cached seed @ …", per-fold metrics, pooled OOF AUC, and cached full-data AUC (when requested). Return OOF.
    Else:
      • Run outer CV with inner tuning, persist fold artifacts + consolidated metrics.json + OOF + signature.
      • Optionally refit on full data once, store + log full-data AUC, update metrics.json.
      • Return OOF.
    """
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)

    N = len(X_train)
    groups = X_train.index
    knobs = {
        "K_outer": K_outer,
        "K_max_inner": K_max_inner,
        "K_stop_inner": K_stop_inner,
        "n_optuna_trials": n_optuna_trials,
    }

    # ===== fast path: cache =====
    if _has_full_cv_cache(
        root, model_type, K_outer, N,
        X_train, y_train, seed,
        K_max_inner, K_stop_inner, n_optuna_trials,
    ):
        print(f"[{model_type}] Using cached seed @ {root}")
        oof, fold_hps, val_aucs, best_iters, fold_metrics, auc_oof, auc_full = _load_cv_cache(
            root, model_type, K_outer
        )

        # Print metrics exactly as training would
        for m in fold_metrics:
            k = m["fold"]
            print(f"[{model_type}] Fold {k}/{K_outer}: TRAIN {m['train_auc']:.4f} | VAL {m['val_auc']:.4f}")
        print(f"[{model_type}] OOF AUC = {auc_oof:.4f}")
        if full_refit and (auc_full is not None):
            print(f"[{model_type}] Full-data TRAIN AUC = {auc_full:.4f}")
        return oof

    # ===== fresh run =====
    oof = np.zeros(N, dtype=np.float32)
    fold_hps: List[dict] = []
    val_aucs: List[float] = []
    best_iters: List[int] = []
    fold_metrics: List[dict] = []

    for k, (tr_idx, va_idx) in enumerate(_outer_cv(K_outer, seed).split(X_train, y_train, groups), 1):
        X_tr, y_tr, g_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx], groups[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        inner = _inner_cv(X_tr, y_tr, g_tr, seed=seed, fold_idx=k - 1,
                          K_max_inner=K_max_inner, K_stop_inner=K_stop_inner)

        pipe, hp, auc_tr, auc_va, best_iter = _tune_and_fit_one_fold(
            model_type, X_tr, y_tr, g_tr, X_va, y_va, inner, n_optuna_trials
        )

        oof[va_idx] = pipe.predict_proba(X_va)[:, 1]
        print(f"[{model_type}] Fold {k}/{K_outer}: TRAIN {auc_tr:.4f} | VAL {auc_va:.4f}")

        if best_iter is not None: best_iters.append(int(best_iter))
        fold_hps.append(hp)
        val_aucs.append(auc_va)
        fold_metrics.append({"fold": k, "train_auc": auc_tr, "val_auc": auc_va})
        _save_fold(root, k - 1, pipe, hp)

    # pooled OOF
    auc_oof = float(roc_auc_score(y_train, oof))
    print(f"[{model_type}] OOF AUC = {auc_oof:.4f}")

    # persist OOF + signature + consolidated metrics (no full-data AUC yet)
    np.save(root / f"oof.npy", oof)
    sig, sig_md5 = _fingerprint_base(X_train, y_train, model_type, seed, params=knobs)
    _save_base_cache_meta(root, model_type, sig, sig_md5)
    _save_base_metrics(root, fold_metrics, auc_oof, auc_train_full=None)

    # optional full refit (single pass)
    if full_refit:
        hp_final = _select_full_hps(full_hp_selection, model_type, fold_hps, val_aucs)
        if best_iters:
            hp_final = hp_final.copy()
            hp_final["n_estimators"] = int(np.median(best_iters))
        if "cat" in model_type:
            hp_final.setdefault("random_seed", seed)
        else:
            hp_final.setdefault("random_state", seed)

        pipe_full = make_pipeline(model_type, hp_final)
        pipe_full.fit(X_train, y_train)
        dump(pipe_full, root / f"full_data_refit.joblib")

        auc_full = float(roc_auc_score(y_train, pipe_full.predict_proba(X_train)[:, 1]))
        print(f"[{model_type}] Full-data TRAIN AUC = {auc_full:.4f}")

        # update consolidated metrics with the full-data AUC
        _save_base_metrics(root, fold_metrics, auc_oof, auc_train_full=auc_full)

    return oof
