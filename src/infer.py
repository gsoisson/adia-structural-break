# src/infer.py

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
from joblib import load
import xgboost as xgb

from src.config import BASE_LEARNERS, MODEL_DIR, INFERENCE_MODE
from src.feature_extraction import build_features
from src.stacking import _stack_features


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────

_FOLD_RE = re.compile(r"fold_(\d+)\.joblib$")


def _best_seeds(base_dir: Path) -> list[int]:
    """Parse best_top*.txt and return the listed seeds (sorted)."""
    txt = sorted(base_dir.glob("best_top*.txt"))[-1]
    seeds = []
    for line in txt.read_text().splitlines():
        if line.startswith("seed="):
            seeds.append(int(line.split("seed=")[1].split()[0]))
    return sorted(seeds)


def _load_fold_pipelines(seed_root: Path) -> list:
    """Load all fold_{k}.joblib under a seed path, sorted by k."""
    fold_dir = seed_root / "fold_models"
    folds = []
    for p in sorted(
        fold_dir.glob("fold_*.joblib"),
        key=lambda q: int(_FOLD_RE.search(q.name).group(1)),
    ):
        folds.append(load(p))
    return folds


def _load_base_fold_models(
    model_dir: Path, base_learners: Iterable[str]
) -> dict[str, dict[int, list]]:
    """
    Returns:
      { model_type: { seed: [fold_pipeline0, fold_pipeline1, ...] } }
    """
    out: dict[str, dict[int, list]] = {}
    base_root = model_dir / "base"
    for m in base_learners:
        m_root = base_root / m
        seed_map: dict[int, list] = {}
        for s in _best_seeds(m_root):
            folds = _load_fold_pipelines(m_root / f"seed_{s}")
            seed_map[s] = folds
        out[m] = seed_map
    return out


def _load_base_full_models(
    model_dir: Path, base_learners: Iterable[str]
) -> dict[str, dict[int, object]]:
    """
    Returns:
      { model_type: { seed: full_refit_pipeline } }
    Expects a file named {model_type}_full.joblib under each seed directory.
    """
    out: dict[str, dict[int, object]] = {}
    base_root = model_dir / "base"
    for m in base_learners:
        m_root = base_root / m
        seed_map: dict[int, object] = {}
        for s in _best_seeds(m_root):
            p = m_root / f"seed_{s}" / f"{m}_full.joblib"
            seed_map[s] = load(p)
        out[m] = seed_map
    return out


def _load_meta_model_paths(model_dir: Path) -> list[str]:
    """Load meta_artifact.joblib and return list of XGB model paths."""
    art = load(model_dir / "meta_artifact.joblib")
    return list(art["model_paths"])


# ─────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────


def _predict_base_fold_ensemble(fold_pipes: list, X: pd.DataFrame) -> np.ndarray:
    """Average predictions across all folds for ONE seed."""
    preds = [
        pipe.predict_proba(X)[:, 1].astype(np.float32, copy=False)
        for pipe in fold_pipes
    ]
    return np.mean(np.vstack(preds), axis=0).astype(np.float32, copy=False)


def _predict_base_across_seeds_fold(
    models_per_seed: dict[int, list], X: pd.DataFrame
) -> np.ndarray:
    """For one base (fold mode): average across folds per seed, then average across seeds."""
    per_seed = [
        _predict_base_fold_ensemble(folds, X) for folds in models_per_seed.values()
    ]
    return np.mean(np.vstack(per_seed), axis=0).astype(np.float32, copy=False)


def _predict_base_across_seeds_full(
    models_per_seed: dict[int, object], X: pd.DataFrame
) -> np.ndarray:
    """For one base (full-refit mode): average full-refit pipelines across seeds."""
    per_seed = [
        pipe.predict_proba(X)[:, 1].astype(np.float32, copy=False)
        for pipe in models_per_seed.values()
    ]
    return np.mean(np.vstack(per_seed), axis=0).astype(np.float32, copy=False)


def _predict_meta_xgb(model_paths: list[str], F_star: pd.DataFrame) -> np.ndarray:
    """Average predictions across the meta top seeds (all XGBoost)."""
    dtest = xgb.DMatrix(F_star.values, feature_names=list(F_star.columns))
    preds = []
    for mp in model_paths:
        booster = xgb.Booster()
        booster.load_model(mp)
        preds.append(booster.predict(dtest).astype(np.float32, copy=False))
    return np.mean(np.vstack(preds), axis=0).astype(np.float32, copy=False)


# alias to avoid mypy complaints if xgboost is imported as xgb
xgboost = xgb

# ─────────────────────────────────────────────────────────────────────
# iinfer(): choose between fold-ensemble or full-refit bases
# ─────────────────────────────────────────────────────────────────────


def infer(
    X_test: Iterable[pd.DataFrame],
    model_dir: Path = MODEL_DIR,
    inference_mode: str = INFERENCE_MODE,  # "fold" | "full"
    base_learners: list = BASE_LEARNERS,
):
    """
    Inference pipeline:

    inference_mode = "fold" (default):
        • Load per-seed fold models for each base.
        • For each shard (one id):
            - build per-id features (inference=True),
            - base preds: average across folds per seed, then across seeds,
            - stack → meta features → XGB meta (avg across meta top seeds),
            - yield np.ndarray([prob]).

    inference_mode = "full":
        • Load per-seed FULL-REFIT pipelines for each base ({model}_full.joblib).
        • Same as above but base preds are averaged across seeds only (no folds).
    """
    if inference_mode == "fold":
        base_models = _load_base_fold_models(model_dir, base_learners)
        base_predict = _predict_base_across_seeds_fold
    elif inference_mode == "full":
        base_models = _load_base_full_models(model_dir, base_learners)
        base_predict = _predict_base_across_seeds_full
    else:
        raise ValueError("inference_mode must be 'fold' or 'full'")

    meta_paths = _load_meta_model_paths(model_dir)

    # signal readiness
    yield

    # process each dataset exactly once (each shard = one id's long df)
    for df_raw in X_test:
        # one id per shard
        id1 = df_raw.index.get_level_values("id").unique()[0]

        # per-id features (no cache I/O)
        X_feat = build_features(df_raw, inference=True).reindex([id1])

        # base predictions → stacked frame
        base_stack = {
            f"{m}_oof": base_predict(models_per_seed, X_feat)
            for m, models_per_seed in base_models.items()
        }
        S_star = pd.DataFrame(base_stack, index=[id1])

        # meta features + XGB meta prediction
        F_star = _stack_features(S_star)
        pred = _predict_meta_xgb(meta_paths, F_star)

        yield pred.astype(np.float32, copy=False)
