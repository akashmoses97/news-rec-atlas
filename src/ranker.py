"""
ranker.py
---------
Train logistic regression on a feature matrix and evaluate ranking quality.

Metrics (per CP2 Section 4 — Metrics Plan):
  - AUC     : overall discrimination (full population)
  - MRR     : mean reciprocal rank (how early the first click appears)
  - NDCG@10 : position-aware ranking quality at top-10
  - HR@10   : hit rate at top-10 (fraction of impressions with >=1 click in top-10)

All metrics return NaN (not crash) when a slice has only one class -- this is expected
for small cold-start slices where some val impressions have no clicks at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


KEYS = ["impression_id", "candidate_id"]


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def mrr_score(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray) -> float:
    """Mean Reciprocal Rank. Impressions with no clicks are skipped."""
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    rrs = []
    for _, sub in df.groupby("g", sort=False):
        if sub["y"].sum() == 0:
            continue
        order = sub.sort_values("s", ascending=False)
        ranks = np.arange(1, len(order) + 1)
        first_hit = ranks[order["y"].values == 1]
        if first_hit.size:
            rrs.append(1.0 / first_hit[0])
    return float(np.mean(rrs)) if rrs else float("nan")


def ndcg_at_k(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 10
) -> float:
    """
    NDCG@k with binary relevance. Impressions with no clicks are skipped.
    """
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    ndcgs = []
    for _, sub in df.groupby("g", sort=False):
        if sub["y"].sum() == 0:
            continue
        order = sub.sort_values("s", ascending=False).head(k)
        gains = order["y"].values
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float((gains * discounts).sum())
        ideal_gains = np.sort(sub["y"].values)[::-1][:k]
        idcg = float((ideal_gains * (1.0 / np.log2(
            np.arange(2, len(ideal_gains) + 2)))).sum())
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def hit_rate_at_k(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 10
) -> float:
    """
    HR@k: fraction of impressions where >=1 clicked article is in top-k.
    Per CP2 Section 4: 'HR@K measures presence of relevant items in top-K results.'
    """
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    hits = []
    for _, sub in df.groupby("g", sort=False):
        if sub["y"].sum() == 0:
            continue
        top_k = sub.sort_values("s", ascending=False).head(k)
        hits.append(int(top_k["y"].sum() > 0))
    return float(np.mean(hits)) if hits else float("nan")


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC. Returns NaN when only one class is present.
    WHY: cold-start val slices can have impressions where every candidate is
    non-clicked, making AUC undefined. NaN is the correct signal -- not a crash.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ------------------------------------------------------------------
# Feature merging
# ------------------------------------------------------------------

def _to_list(x) -> List[pd.DataFrame]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def merge_features(
    long_df: pd.DataFrame,
    feature_dfs: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """
    Left-join long_df with feature DataFrames on (impression_id, candidate_id).
    Missing values are 0-filled: honest representation of absent signal.
    WHY 0-fill: a cold-start user has 0 neighbor overlap, not an unknown amount.
    The model learns the weight of 'no signal' vs 'some signal'.
    """
    base_cols = [
        "impression_id", "candidate_id", "user_id",
        "history_len", "label", "position", "is_cold_start",
    ]
    merged = long_df[base_cols].copy()
    for f in feature_dfs:
        feat_cols = [c for c in f.columns if c not in KEYS]
        merged = merged.merge(f[KEYS + feat_cols], on=KEYS, how="left")
    all_feat_cols = [c for c in merged.columns if c not in base_cols]
    merged[all_feat_cols] = merged[all_feat_cols].fillna(0.0)
    return merged


# ------------------------------------------------------------------
# Training & evaluation
# ------------------------------------------------------------------

@dataclass
class ContestantResult:
    name: str
    slice: str
    n_impressions: int
    n_clicks: int
    auc: float
    mrr: float
    ndcg10: float
    hr10: float      
    n_features: int


def _eval_slice(
    val_df: pd.DataFrame,
    scores: np.ndarray,
) -> Tuple[int, int, float, float, float, float]:
    """
    Return (n_imp, n_clicks, auc, mrr, ndcg10, hr10). All NaN-safe.
    """
    y = val_df["label"].values
    g = val_df["impression_id"].values
    return (
        val_df["impression_id"].nunique(),
        int(y.sum()),
        safe_auc(y, scores),
        mrr_score(y, scores, g),
        ndcg_at_k(y, scores, g, k=10),
        hit_rate_at_k(y, scores, g, k=10),
    )


def train_lr(
    train_merged: pd.DataFrame,
    feature_cols: List[str],
    C: float = 1.0,
) -> LogisticRegression:
    """
    Pointwise logistic regression.
    class_weight='balanced' corrects for the ~4% click rate imbalance.
    WHY LR: per CP2 design, holding model class fixed so metric differences
    are attributable to features, not model capacity.
    """
    X = train_merged[feature_cols].values
    y = train_merged["label"].values
    model = LogisticRegression(
        C=C, solver="liblinear", max_iter=200,
        class_weight="balanced", random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate(
    model: LogisticRegression,
    val_merged: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Dict]:
    """
    Score val rows; return metrics for full population and cold-start slice.
    """
    scores = model.predict_proba(val_merged[feature_cols].values)[:, 1]
    val_scored = val_merged.assign(_score=scores)

    full_stats = _eval_slice(val_scored, val_scored["_score"].values)

    # Use is_cold_start flag if present (single-visit users ~33%),
    # fall back to history_len == 0 if not.
    if "is_cold_start" in val_scored.columns:
        cold_df = val_scored[val_scored["is_cold_start"]]
    else:
        cold_df = val_scored[val_scored["history_len"] == 0]

    cold_stats = (
        _eval_slice(cold_df, cold_df["_score"].values)
        if len(cold_df) > 0
        else (0, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    )

    def _pack(n_imp, n_clk, auc, mrr, ndcg, hr):
        return dict(n_impressions=n_imp, n_clicks=n_clk,
                    auc=auc, mrr=mrr, ndcg10=ndcg, hr10=hr)

    return {
        "full":       _pack(*full_stats),
        "cold_start": _pack(*cold_stats),
    }


# ------------------------------------------------------------------
# Contest harness
# ------------------------------------------------------------------

def run_contest(
    contestants: Dict[str, Tuple[Sequence[pd.DataFrame], Sequence[pd.DataFrame]]],
    train_long: pd.DataFrame,
    val_long: pd.DataFrame,
    C: float = 1.0,
) -> pd.DataFrame:
    """
    Train + evaluate each contestant. Returns tidy results DataFrame.

    contestants : name -> (train_feature_dfs, val_feature_dfs)
    """
    rows: List[ContestantResult] = []

    for name, (train_feats, val_feats) in contestants.items():
        train_feats = _to_list(train_feats)
        val_feats   = _to_list(val_feats)

        train_merged = merge_features(train_long, train_feats)
        val_merged   = merge_features(val_long,   val_feats)

        feature_cols = [
            c for c in train_merged.columns
            if c not in {
                "impression_id", "candidate_id", "user_id",
                "history_len", "label", "position",
            }
        ]
        if not feature_cols:
            raise ValueError(f"Contestant {name!r}: no feature columns found.")

        # Include position as a display-order control feature
        if "position" not in feature_cols:
            feature_cols = ["position"] + feature_cols

        model   = train_lr(train_merged, feature_cols, C=C)
        metrics = evaluate(model, val_merged, feature_cols)

        for slice_name, m in metrics.items():
            rows.append(ContestantResult(
                name=name, slice=slice_name,
                n_impressions=m["n_impressions"],
                n_clicks=m["n_clicks"],
                auc=m["auc"], mrr=m["mrr"],
                ndcg10=m["ndcg10"], hr10=m["hr10"],
                n_features=len(feature_cols),
            ))

    return pd.DataFrame([r.__dict__ for r in rows])


def scoreboard(results: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """
    Wide scoreboard table for one slice. NaN preserved (not coerced to 0).
    """
    sub = (
        results[results["slice"] == slice_name]
        .copy()
        .set_index("name")
        [["auc", "mrr", "ndcg10", "hr10",
          "n_impressions", "n_clicks", "n_features"]]
    )
    return sub.round(4)