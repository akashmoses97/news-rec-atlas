"""
utils.py
--------
Shared utilities: chronological train/val split, cold-start masking, RNG seeding.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import pandas as pd


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed numpy and python RNGs. Call at the start of the notebook."""
    random.seed(seed)
    np.random.seed(seed)


def chronological_split(
    behaviors: pd.DataFrame,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting impressions chronologically.

    The most recent `val_frac` fraction of impressions go to validation;
    the rest are training. This mimics deployment: train on the past, evaluate
    on the future. Avoids the leakage which might happen with random splits.

    Returns
    -------
    train_df, val_df : pd.DataFrame
        Same columns as input, sorted by time, disjoint by row.
    """
    if "time" not in behaviors.columns:
        raise ValueError("behaviors must have a 'time' column")

    sorted_df = behaviors.sort_values("time", kind="stable").reset_index(drop=True)
    cut = int(len(sorted_df) * (1 - val_frac))
    train_df = sorted_df.iloc[:cut].copy()
    val_df = sorted_df.iloc[cut:].copy()
    return train_df, val_df


def cold_start_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for cold-start users (single-impression users).
    Matches CP1 EDA definition: ~32.77% of users appear only once.
    Uses is_cold_start flag if pre-computed, otherwise falls back to history_len == 0.
    """
    if "is_cold_start" in df.columns:
        return df["is_cold_start"].astype(bool)
    if "history_len" not in df.columns:
        raise ValueError("df must have 'is_cold_start' or 'history_len' column")
    return df["history_len"] == 0


def describe_split(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiny summary table for the notebook's setup section."""
    def stats(d, name):
        cold = d['is_cold_start'].sum() if 'is_cold_start' in d.columns \
           else cold_start_mask(d).sum()
        return {
            "split": name,
            "impressions": len(d),
            "users": d["user_id"].nunique(),
            "cold_start_impressions": int(cold),
            "cold_start_pct": round(cold / max(len(d), 1) * 100, 2),
        }
    return pd.DataFrame([stats(train_df, "train"), stats(val_df, "val")])