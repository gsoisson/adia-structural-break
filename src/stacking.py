# ─────────────────────────────────────────────────────────────────────
# 0) Imports & config guard
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    BASE_LEARNERS,
    FULL_HP_SELECTION,
    FULL_REFIT,
    K_MAX_INNER,
    K_OUTER,
    K_STOP_INNER,
    MODEL_DIR,
    N_OPTUNA_TRIALS,
    N_SEEDS,
    RANDOM_STATE,
    TOP_SEEDS,
)
from src.feature_extraction import build_features
from src.train import train_model

assert 1 <= TOP_SEEDS <= N_SEEDS, "TOP_SEEDS must be in [1, N_SEEDS]"


# ─────────────────────────────────────────────────────────────────────
# 1) Console banners
# ─────────────────────────────────────────────────────────────────────

def _print_base_header(model_type: str, width: int = 72) -> None:
    line = "═" * width
    title = f" TRAINING BASE — {model_type} "
    print("\n" + line)
    print(title.center(width, "═"))
    print(line)


def _print_seed_header(model_type: str, seed: int, idx: int, total: int, width: int = 72) -> None:
    print("\n" + "─" * width)
    print(f"[{model_type}] SEED {seed} ({idx}/{total}) — START")


def _print_auc_highlight(scope: str, auc: float, top_k: int, width: int = 72) -> None:
    bar = "█" * width
    msg = f"{scope} — TOP{top_k} avg OOF AUC = {auc:.4f}"
    print("\n" + bar)
    print(msg.center(width))
    print(bar)


def _print_meta_header(model_type: str, width: int = 72) -> None:
    line = "═" * width
    title = f" TRAINING META — {model_type} "
    print("\n" + line)
    print(title.center(width, "═"))
    print(line)


# ─────────────────────────────────────────────────────────────────────
# 2) Small utilities
# ─────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _seed_list(seed0: int, n: int) -> list[int]:
    return [int(seed0 + 17 * i) for i in range(n)]


def _avg_topk(records: List[dict], k: int) -> Tuple[np.ndarray, List[dict]]:
    """Return averaged OOF of top-k records (by 'auc') and the top-k records themselves."""
    ranked = sorted(records, key=lambda r: r["auc"], reverse=True)[: max(1, k)]
    avg = np.mean(np.vstack([r["oof"] for r in ranked]), axis=0)
    return avg, ranked


# ─────────────────────────────────────────────────────────────────────
# 3) Base learners (multi-seed orchestration)
# ─────────────────────────────────────────────────────────────────────

def _train_base_one_seed(
    X_feat: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    seed: int,
    root: Path,
    K_outer: int = K_OUTER,
    K_max_inner: int = K_MAX_INNER,
    K_stop_inner: int = K_STOP_INNER,
    n_optuna_trials: int = N_OPTUNA_TRIALS,
    full_refit: bool = FULL_REFIT,
    full_hp_selection: str = FULL_HP_SELECTION,
) -> dict:
    """Train one seed for a base learner under root/base/<model_type>/seed_<seed>/."""
    seed_dir = _ensure_dir(root / "base" / model_type / f"seed_{seed}")
    oof = train_model(
        X_feat,
        y,
        model_type=model_type,
        K_outer=K_outer,
        K_max_inner=K_max_inner,
        K_stop_inner=K_stop_inner,
        n_optuna_trials=n_optuna_trials,
        seed=seed,
        model_dir=str(seed_dir),
        full_refit=full_refit,
        full_hp_selection=full_hp_selection,
    )
    np.save(seed_dir / "oof.npy", oof.astype(np.float32))
    auc = float(roc_auc_score(y, oof))
    return {"seed": seed, "oof": oof, "auc": auc, "dir": str(seed_dir)}


def _train_base_multi_seeds(
    X_feat: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    root: Path,
    seed0: int,
    n_seeds: int = N_SEEDS,
    top_seeds: int = TOP_SEEDS,
) -> dict:
    """Train N_SEEDS; keep top_seeds by OOF AUC; save averaged OOF + human summary."""
    model_root = _ensure_dir(root / "base" / model_type)
    records: List[dict] = []

    for idx, s in enumerate(_seed_list(seed0, n_seeds), 1):
        _print_seed_header(model_type, s, idx, n_seeds)
        records.append(_train_base_one_seed(X_feat, y, model_type, s, root))

    avg_oof, top = _avg_topk(records, top_seeds)
    np.save(model_root / f"avg_top{top_seeds}_oof.npy", avg_oof.astype(np.float32))

    # Ensemble AUC of the averaged predictions
    auc_ensemble = float(roc_auc_score(y, avg_oof))

    # Write avg_top{K}.txt exactly as requested
    lines = "\n".join(f"seed={r['seed']} auc={r['auc']:.6f} dir={r['dir']}" for r in top)
    (model_root / f"avg_top{top_seeds}.txt").write_text(f"{lines}\nAVG_AUC={auc_ensemble:.6f}\n")

    _print_auc_highlight(f"[BASE {model_type}]", auc_ensemble, top_seeds)
    return {"model_type": model_type, "avg_oof": avg_oof, "auc": auc_ensemble, "top": top, "root": str(model_root)}


# ─────────────────────────────────────────────────────────────────────
# 4) Meta-learner features (stack)
# ─────────────────────────────────────────────────────────────────────

def _stack_features(S_base: pd.DataFrame) -> pd.DataFrame:
    """Meta features = per-base logits + rowwise mean/std of logits."""
    def _logit_clip(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        p = np.clip(p.astype(np.float32), eps, 1 - eps)
        return np.log(p) - np.log(1.0 - p)

    L = pd.DataFrame({f"{c}_logit": _logit_clip(S_base[c].values) for c in S_base.columns}, index=S_base.index)
    stats = pd.DataFrame(
        {"logit_mean": L.mean(axis=1).astype(np.float32), "logit_std": L.std(axis=1).astype(np.float32)},
        index=S_base.index,
    )
    return pd.concat([L, stats], axis=1)


# ─────────────────────────────────────────────────────────────────────
# 5) Meta-learner cache (signature & metrics)
# ─────────────────────────────────────────────────────────────────────

def _fingerprint_meta(F: pd.DataFrame, y: pd.Series, seed: int, params: dict, K_outer: int) -> tuple[dict, str]:
    sig = {
        "seed": int(seed),
        "K_outer": int(K_outer),
        "n_rows": int(len(F)),
        "n_cols": int(F.shape[1]),
        "cols": list(F.columns),
        "params": params,
        "y_sum": int(np.asarray(y).sum()),
        "y_len": int(len(y)),
    }
    raw = json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sig, hashlib.md5(raw).hexdigest()


def _save_meta_signature(out_dir: Path, sig: dict, sig_hash: str) -> None:
    (out_dir / "meta.json").write_text(json.dumps({"signature": sig, "signature_md5": sig_hash}, indent=2))


def _load_meta_signature(out_dir: Path) -> dict:
    p = out_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_meta_metrics(
    out_dir: Path,
    fold_metrics: list[dict],
    auc_oof: float,
    auc_train_full: float,
) -> None:
    payload = {
        "fold_metrics": fold_metrics,          # [{fold, train_auc, val_auc}]
        "auc_oof": float(auc_oof),             # pooled OOF AUC
        "auc_train_full": float(auc_train_full),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def _load_meta_metrics(out_dir: Path) -> dict:
    p = out_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────
# 6) Meta-learner CV + fit (single seed)
# ─────────────────────────────────────────────────────────────────────

def _meta_xgb_oof(
    F: pd.DataFrame,
    y: pd.Series,
    seed: int,
    params: dict,
    K_outer: int = K_OUTER,
) -> tuple[float, np.ndarray, list[dict]]:
    """Compute meta OOF preds + per-fold metrics using StratifiedGroupKFold."""
    yv = y.astype(int).to_numpy()
    groups = F.index.to_numpy()
    skf = StratifiedGroupKFold(n_splits=K_outer, shuffle=True, random_state=seed)

    oof = np.zeros(len(F), dtype=np.float32)
    cols = F.columns.tolist()
    fold_metrics: list[dict] = []

    for k, (tr_idx, va_idx) in enumerate(skf.split(F, yv, groups), 1):
        Ftr, Fva = F.iloc[tr_idx], F.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        hp = params.copy()
        num_round = int(hp.pop("n_estimators", 200))
        dtr = xgb.DMatrix(Ftr.values, label=ytr.values, feature_names=cols)
        dva = xgb.DMatrix(Fva.values, label=yva.values, feature_names=cols)
        booster = xgb.train(params=hp, dtrain=dtr, num_boost_round=num_round, verbose_eval=False)

        p_tr = booster.predict(dtr).astype(np.float32)
        p_va = booster.predict(dva).astype(np.float32)
        oof[va_idx] = p_va

        auc_tr = float(roc_auc_score(ytr, p_tr))
        auc_va = float(roc_auc_score(yva, p_va))
        fold_metrics.append({"fold": k, "train_auc": auc_tr, "val_auc": auc_va})
        print(f"[META-XGB] Fold {k}/{K_outer}: TRAIN {auc_tr:.4f} | VAL {auc_va:.4f}")

    auc_oof = float(roc_auc_score(y, oof))
    print(f"[META-XGB] OOF AUC = {auc_oof:.4f}")
    return auc_oof, oof, fold_metrics


def _train_meta_xgb_one_seed(
    S_base: pd.DataFrame,
    y: pd.Series,
    seed: int,
    out_dir: Path,
    params: dict = None,
    K_outer: int = K_OUTER,
) -> dict:
    """
    Save (per seed):
      • meta.json       (signature only)
      • metrics.json    ({fold_metrics, auc_oof, auc_train_full})
      • oof.npy
      • model.json      (booster on all data)
    If signature matches and files exist → print cached metrics (like training) and return.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    F = _stack_features(S_base)

    # default meta-XGB params (tiny tree)
    hp = dict(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=2,
        learning_rate=0.05,
        n_estimators=200,       # used as num_boost_round
        subsample=0.7,
        colsample_bytree=1.0,
        min_child_weight=20.0,
        gamma=2.0,
        reg_lambda=20.0,
        reg_alpha=0.0,
        max_bin=64,
        tree_method="hist",
        verbosity=0,
        seed=seed,
    )
    if params:
        hp.update(params)

    # cache check
    sig, sig_hash = _fingerprint_meta(F, y, seed, hp, K_outer)
    sig_cached = _load_meta_signature(out_dir)
    metrics_cached = _load_meta_metrics(out_dir)
    oof_path = out_dir / "oof.npy"
    model_path = out_dir / "model.json"

    if sig_cached and sig_cached.get("signature_md5") == sig_hash and metrics_cached and oof_path.exists() and model_path.exists():
        print(f"[META-XGB] Using cached seed @ {out_dir}")
        for m in metrics_cached["fold_metrics"]:
            k = m["fold"]
            print(f"[META-XGB] Fold {k}/{K_outer}: TRAIN {m['train_auc']:.4f} | VAL {m['val_auc']:.4f}")
        print(f"[META-XGB] OOF AUC = {metrics_cached['auc_oof']:.4f}")
        print(f"[META-XGB] Full-data TRAIN AUC = {metrics_cached['auc_train_full']:.4f}")

        oof = np.load(oof_path)
        return {"seed": seed, "auc": float(metrics_cached["auc_oof"]), "oof": oof, "dir": str(out_dir)}

    # fresh OOF
    auc_oof, oof, fold_metrics = _meta_xgb_oof(F, y, seed=seed, params=hp, K_outer=K_outer)

    # final model on all data
    hp_full = hp.copy()
    num_round = int(hp_full.pop("n_estimators", 200))
    dtrain = xgb.DMatrix(F.values, label=y.values, feature_names=F.columns.tolist())
    booster = xgb.train(params=hp_full, dtrain=dtrain, num_boost_round=num_round, verbose_eval=False)

    # full-data train AUC
    dfull = xgb.DMatrix(F.values, feature_names=F.columns.tolist())
    p_full = booster.predict(dfull).astype(np.float32)
    auc_full = float(roc_auc_score(y, p_full))
    print(f"[META-XGB] Full-data TRAIN AUC = {auc_full:.4f}")

    # persist
    booster.save_model(str(model_path))
    np.save(oof_path, oof)
    _save_meta_signature(out_dir, sig, sig_hash)
    _save_meta_metrics(out_dir, fold_metrics, auc_oof, auc_full)

    return {"seed": seed, "auc": auc_oof, "oof": oof, "dir": str(out_dir)}


# ─────────────────────────────────────────────────────────────────────
# 7) Meta-learner (multi-seed aggregation)
# ─────────────────────────────────────────────────────────────────────

def _train_meta_xgb_multi_seeds(
    S_base: pd.DataFrame,
    y: pd.Series,
    root: Path,
    seed0: int,
    n_seeds: int = N_SEEDS,
    top_seeds: int = TOP_SEEDS,
) -> dict:
    """Train multiple meta seeds; keep top_seeds; save averaged OOF + artifact + summary."""
    base_dir = _ensure_dir(root / "meta" / "xgb")

    records: List[dict] = []
    seeds = _seed_list(seed0, n_seeds)
    for idx, s in enumerate(seeds, 1):
        _print_seed_header("META-XGB", s, idx, len(seeds))
        records.append(_train_meta_xgb_one_seed(S_base, y, s, base_dir / f"seed_{s}"))

    # averaged meta-OFF over top seeds + ensemble AUC
    avg_oof, top = _avg_topk(records, top_seeds)
    np.save(base_dir / f"avg_top{top_seeds}_oof.npy", avg_oof.astype(np.float32))
    auc_ensemble = float(roc_auc_score(y, avg_oof))

    # human-readable summary
    lines = "\n".join(f"seed={r['seed']} auc={r['auc']:.6f} dir={r['dir']}" for r in top)
    (base_dir / f"avg_top{top_seeds}.txt").write_text(f"{lines}\nAVG_AUC={auc_ensemble:.6f}\n")

    # compact artifact for inference
    artifact = {
        "type": "xgb_meta",
        "top_seeds": [r["seed"] for r in top],
        "model_paths": [str(base_dir / f"seed_{r['seed']}" / "model.json") for r in top],
    }
    dump(artifact, root / "meta_artifact.joblib")

    _print_auc_highlight("[META-XGB]", auc_ensemble, top_seeds)
    return {"avg_oof": avg_oof, "auc": auc_ensemble, "artifact": artifact, "top": top}


# ─────────────────────────────────────────────────────────────────────
# 8) Orchestrator
# ─────────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_models: tuple[str, ...] = BASE_LEARNERS,
    seed: int = RANDOM_STATE,
    model_dir: str = MODEL_DIR,
) -> dict:
    """
    1) Build per-id features once.
    2) Train each base across N_SEEDS; keep TOP_SEEDS; average their OOFs.
    3) Stack averaged OOF columns → train XGB meta (multi-seed); keep TOP_SEEDS; save artifact.
    """
    root = _ensure_dir(Path(model_dir))

    # features once
    X_feat = build_features(X_train, force_prep=False, force_all=False)

    # sanity: one row per id
    n_ids = X_train.index.get_level_values("id").unique().size
    if n_ids != len(X_feat):
        raise ValueError(
            "train_n != feat_n — delete resources/features (they were computed on a different X_train)."
        )

    # bases
    oof_cols: dict[str, np.ndarray] = {}
    base_info: dict[str, dict] = {}
    for i, m in enumerate(base_models):
        _print_base_header(m)
        res = _train_base_multi_seeds(
            X_feat, y_train, model_type=m, root=root, seed0=seed + 100 * (i + 1)
        )
        oof_cols[f"{m}_oof"] = res["avg_oof"]
        base_info[m] = res

    S_base = pd.DataFrame(oof_cols, index=X_feat.index)

    # meta (XGB only)
    _print_meta_header("XGB")
    meta_res = _train_meta_xgb_multi_seeds(S_base, y_train, root=root, seed0=seed + 2000)

    print(f"\n[STACK] Final meta OOF AUC = {meta_res['auc']:.4f}")
    return {
        "stack_base": S_base,
        "per_base": base_info,
        "meta_oof": meta_res["avg_oof"],
        "meta_auc": meta_res["auc"],
        "meta_artifact": meta_res["artifact"],
    }
