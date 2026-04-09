"""
Prediction for an upcoming matchup using latest rolling features prior to a date.
"""

import pickle
from pathlib import Path

import pandas as pd

from src.data_fetch import fetch_2025_26_regular_season
from src.features import (
    DIFF_FEATURE_NAMES,
    DEFAULT_MIN_PERIODS,
    DEFAULT_ROLLING_WINDOW,
    build_team_rolling_features,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
COLUMNS_PATH = Path(__file__).resolve().parent.parent / "feature_columns.pkl"


def predict_matchup(
    home_team_abbr: str,
    away_team_abbr: str,
    as_of_date: str | None = None,
    model=None,
    feature_columns: list | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> float:
    """
    Get latest rolling features for each team prior to as_of_date (or latest game date),
    build DIFF_* in the same order as training, return home-win probability (predict_proba).

    Returns:
        Probability that home team wins (float in [0, 1]).
    """
    home_team_abbr = home_team_abbr.strip().upper()
    away_team_abbr = away_team_abbr.strip().upper()

    games, team_log = fetch_2025_26_regular_season()
    if games.empty or team_log.empty:
        raise ValueError("No 2025-26 regular season data available.")

    team_log = team_log.copy()
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        team_log = team_log[team_log["GAME_DATE"] < as_of]
        if team_log.empty:
            raise ValueError(f"No games before as_of_date={as_of_date}.")
    # else: use all games; "latest" will be the last game in the log

    team_rolling = build_team_rolling_features(
        team_log, rolling_window=rolling_window, min_periods=min_periods
    )
    if team_rolling.empty:
        raise ValueError("Could not build rolling features (no data after filtering).")

    # Latest pre-game rolling row per team (last row per team after sort)
    team_rolling = team_rolling.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
    home_rows = team_rolling[team_rolling["TEAM_ABBREVIATION"] == home_team_abbr]
    away_rows = team_rolling[team_rolling["TEAM_ABBREVIATION"] == away_team_abbr]

    if home_rows.empty:
        raise ValueError(f"No games found for home team '{home_team_abbr}' before as_of_date.")
    if away_rows.empty:
        raise ValueError(f"No games found for away team '{away_team_abbr}' before as_of_date.")

    home_last = home_rows.iloc[-1]
    away_last = away_rows.iloc[-1]

    # Build DIFF_* in same order as training
    diff_row = {}
    for col in ["PTS", "REB", "AST", "TOV", "PLUS_MINUS"]:
        hkey = f"AVG_{col}"
        akey = f"AVG_{col}"
        if hkey not in home_last or akey not in away_last:
            raise ValueError(f"Missing rolling column AVG_{col} for matchup.")
        diff_row[f"DIFF_{col}"] = home_last[hkey] - away_last[akey]

    X = pd.DataFrame([diff_row])[DIFF_FEATURE_NAMES]

    # Load model and columns if not provided
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first (e.g. main.py)."
            )
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if feature_columns is None:
        if COLUMNS_PATH.exists():
            with open(COLUMNS_PATH, "rb") as f:
                feature_columns = pickle.load(f)
        else:
            feature_columns = DIFF_FEATURE_NAMES

    # Safety: X columns must match training
    assert list(X.columns) == feature_columns, (
        f"Predict X columns must match training: {list(X.columns)} vs {feature_columns}"
    )

    proba = model.predict_proba(X)[0, 1]
    return float(proba)
