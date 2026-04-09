"""
Train Logistic Regression baseline on 2025-26 NBA game-level DIFF_* features.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split

from src.data_fetch import fetch_2025_26_regular_season
from src.features import DIFF_FEATURE_NAMES, build_X_y

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
COLUMNS_PATH = Path(__file__).resolve().parent.parent / "feature_columns.pkl"


def train_baseline(
    rolling_window: int = 10,
    min_periods: int = 3,
    save_model: bool = True,
) -> tuple[LogisticRegression, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Fetch data, build features, split, fit LogisticRegression, evaluate.
    Returns (model, X_train, y_train, X_test, y_test).
    """
    games, team_log = fetch_2025_26_regular_season()
    if games.empty or team_log.empty:
        raise ValueError("No 2025-26 regular season data returned from nba_api.")

    X, y = build_X_y(
        games, team_log, rolling_window=rolling_window, min_periods=min_periods, drop_na=True
    )
    if X.empty or y.empty:
        raise ValueError("No samples after building features (check rolling/min_periods).")

    # Safety: ensure both classes present
    assert y.nunique() == 2, (
        f"Target must have exactly 2 classes; got {y.nunique()}. "
        f"Class counts: {y.value_counts().to_dict()}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    assert y_train.nunique() == 2, (
        f"Train target must have exactly 2 classes; got {y_train.nunique()}. "
        f"Class counts: {y_train.value_counts().to_dict()}"
    )

    # Align columns (should already match)
    feature_cols = list(X.columns)
    assert feature_cols == DIFF_FEATURE_NAMES, (
        f"X columns must match DIFF_FEATURE_NAMES: {feature_cols} vs {DIFF_FEATURE_NAMES}"
    )

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print("--- Train/Test Split ---")
    print(f"Total samples: {len(X)}")
    print(f"y distribution (overall): {y.value_counts().sort_index().to_dict()}")
    print(f"y_train distribution: {y_train.value_counts().sort_index().to_dict()}")
    print()
    print("--- Evaluation (test set) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Away win", "Home win"]))

    if save_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(COLUMNS_PATH, "wb") as f:
            pickle.dump(feature_cols, f)

    return model, X_train, y_train, X_test, y_test
