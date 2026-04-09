"""
Build team pre-game rolling features (no leakage) and game-level DIFF_* features.
"""

import pandas as pd

ROLLING_STAT_COLUMNS = ["PTS", "REB", "AST", "TOV", "PLUS_MINUS"]
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_MIN_PERIODS = 3

# Final feature names used in model (same order as required for predict)
DIFF_FEATURE_NAMES = [
    "DIFF_PTS",
    "DIFF_REB",
    "DIFF_AST",
    "DIFF_TOV",
    "DIFF_PLUS_MINUS",
]


def build_team_rolling_features(
    team_log: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """
    For each team, build rolling averages using only PRIOR games (shift(1) then rolling).
    team_log: one row per team per game with GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, PTS, REB, AST, TOV, PLUS_MINUS.
    Returns same rows with added columns AVG_PTS, AVG_REB, AVG_AST, AVG_TOV, AVG_PLUS_MINUS (pre-game rolling).
    """
    team_log = team_log.copy()
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])
    for c in ROLLING_STAT_COLUMNS:
        if c not in team_log.columns:
            team_log[c] = pd.to_numeric(team_log[c], errors="coerce")

    # Sort by team and date so order is well-defined
    team_log = team_log.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    out = []
    for team_id, grp in team_log.groupby("TEAM_ID"):
        grp = grp.copy()
        for col in ROLLING_STAT_COLUMNS:
            if col not in grp.columns:
                continue
            # Shift so current game is excluded, then rolling mean over prior games
            grp[f"AVG_{col}"] = (
                grp[col].shift(1).rolling(window=rolling_window, min_periods=min_periods).mean()
            )
        out.append(grp)
    if not out:
        return pd.DataFrame()
    result = pd.concat(out, ignore_index=True)
    return result


def build_game_level_features(
    games: pd.DataFrame,
    team_rolling: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Join games with home/away pre-game rolling features and compute DIFF_* = HOME_AVG - AWAY_AVG.
    games: GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_PTS, AWAY_PTS, ...
    team_rolling: one row per team per game with GAME_ID, TEAM_ID, AVG_PTS, AVG_REB, ...
    """
    # For each game we need the PRE-GAME rolling row for home and away.
    # team_rolling row for (GAME_ID, TEAM_ID) is that team's stats *in* that game; the AVG_* there are pre-game.
    home_roll = team_rolling.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        **{f"AVG_{c}": f"HOME_AVG_{c}" for c in ROLLING_STAT_COLUMNS},
    })
    home_roll = home_roll[["GAME_ID", "HOME_TEAM_ID"] + [f"HOME_AVG_{c}" for c in ROLLING_STAT_COLUMNS]]
    away_roll = team_rolling.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        **{f"AVG_{c}": f"AWAY_AVG_{c}" for c in ROLLING_STAT_COLUMNS},
    })
    away_roll = away_roll[["GAME_ID", "AWAY_TEAM_ID"] + [f"AWAY_AVG_{c}" for c in ROLLING_STAT_COLUMNS]]

    merged = games.merge(home_roll, on=["GAME_ID", "HOME_TEAM_ID"], how="inner")
    merged = merged.merge(away_roll, on=["GAME_ID", "AWAY_TEAM_ID"], how="inner")

    for col in ROLLING_STAT_COLUMNS:
        merged[f"DIFF_{col}"] = merged[f"HOME_AVG_{col}"] - merged[f"AWAY_AVG_{col}"]

    if drop_na:
        merged = merged.dropna(subset=DIFF_FEATURE_NAMES)

    return merged


def get_target(games_with_features: pd.DataFrame) -> pd.Series:
    """TARGET = 1 if home team wins (HOME_PTS > AWAY_PTS), else 0."""
    return (games_with_features["HOME_PTS"] > games_with_features["AWAY_PTS"]).astype(int)


def build_X_y(
    games: pd.DataFrame,
    team_log: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: team rolling -> game-level DIFF_* -> X (only DIFF_*), y (binary home win).
    Returns (X, y) with same index; rows with NaN DIFF_* are dropped if drop_na=True.
    """
    team_rolling = build_team_rolling_features(
        team_log, rolling_window=rolling_window, min_periods=min_periods
    )
    if team_rolling.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    games_with_features = build_game_level_features(games, team_rolling, drop_na=drop_na)
    if games_with_features.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = games_with_features[DIFF_FEATURE_NAMES].copy()
    y = get_target(games_with_features)
    return X, y
