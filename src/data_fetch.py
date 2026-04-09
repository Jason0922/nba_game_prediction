"""
Data collection for 2025-26 NBA regular season via nba_api.
Fetches all games and team game logs (boxscore-derived stats).
"""

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


SEASON = "2025-26"
SEASON_TYPE = "Regular Season"


def fetch_league_game_log(season: str = SEASON, season_type: str = SEASON_TYPE) -> pd.DataFrame:
    """Fetch full league game log (one row per team per game) for the given season."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    df = log.get_data_frames()[0]
    return df


def parse_matchup(matchup: str) -> tuple[str, str]:
    """
    Parse MATCHUP into (away_abbr, home_abbr).
    Formats: 'AWAY @ HOME' (e.g. 'BOS @ ATL') or 'AWAY vs. HOME' (e.g. 'DAL vs. SAS' -> SAS home).
    """
    s = matchup.strip()
    if " @ " in s:
        parts = s.split(" @ ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    # "X vs. Y" or "X vs Y" -> Y is home team
    if " vs. " in s:
        parts = s.split(" vs. ", 1)
    elif " vs " in s:
        parts = s.split(" vs ", 1)
    else:
        raise ValueError(f"Unexpected MATCHUP format: {matchup}")
    if len(parts) != 2:
        raise ValueError(f"Unexpected MATCHUP format: {matchup}")
    return parts[0].strip(), parts[1].strip()


def build_games_table(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per game: GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_PTS, AWAY_PTS.
    MATCHUP is 'AWAY @ HOME', so the row's TEAM_ABBREVIATION is either home or away.
    """
    # Ensure numeric types
    team_log = team_log.copy()
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])
    for col in ["PTS", "REB", "AST", "TOV", "PLUS_MINUS"]:
        if col in team_log.columns:
            team_log[col] = pd.to_numeric(team_log[col], errors="coerce")

    rows = []
    for gid, grp in team_log.groupby("GAME_ID"):
        if len(grp) != 2:
            continue
        grp = grp.sort_values("GAME_DATE").iloc[0:2]
        matchups = grp["MATCHUP"].tolist()
        # Use first row's MATCHUP to get home/away (both rows have same game, so same matchup pattern)
        away_abbr, home_abbr = parse_matchup(grp["MATCHUP"].iloc[0])
        home_row = grp[grp["TEAM_ABBREVIATION"] == home_abbr].iloc[0]
        away_row = grp[grp["TEAM_ABBREVIATION"] == away_abbr].iloc[0]
        rows.append({
            "GAME_ID": gid,
            "GAME_DATE": home_row["GAME_DATE"],
            "HOME_TEAM_ID": home_row["TEAM_ID"],
            "AWAY_TEAM_ID": away_row["TEAM_ID"],
            "HOME_TEAM_ABBREVIATION": home_abbr,
            "AWAY_TEAM_ABBREVIATION": away_abbr,
            "HOME_PTS": home_row["PTS"],
            "AWAY_PTS": away_row["PTS"],
        })
    games = pd.DataFrame(rows)
    if not games.empty:
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        games = games.sort_values("GAME_DATE").reset_index(drop=True)
    return games


def fetch_2025_26_regular_season() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch 2025-26 regular season data.
    Returns:
        games: DataFrame with columns GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID,
               HOME_TEAM_ABBREVIATION, AWAY_TEAM_ABBREVIATION, HOME_PTS, AWAY_PTS.
        team_log: Full league game log (one row per team per game) with PTS, REB, AST, TOV, PLUS_MINUS.
    """
    team_log = fetch_league_game_log(season=SEASON, season_type=SEASON_TYPE)
    if team_log.empty:
        return pd.DataFrame(), team_log
    games = build_games_table(team_log)
    return games, team_log
