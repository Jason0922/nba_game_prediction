"""
Entry script: fetch 2025-26 regular season data, build rolling features,
train Logistic Regression, print evaluation, run example prediction.
"""

from src.data_fetch import fetch_2025_26_regular_season
from src.train import train_baseline
from src.predict import predict_matchup


def main() -> None:
    print("=== NBA Game Outcome Baseline (2025-26 Regular Season) ===\n")

    # 1) Pull data, build features, train, evaluate
    model, X_train, y_train, X_test, y_test = train_baseline(
        rolling_window=10,
        min_periods=3,
        save_model=True,
    )

    # 2) Example prediction: use two team abbreviations from the season data
    games, _ = fetch_2025_26_regular_season()
    if not games.empty:
        abbrs = list(
            games["HOME_TEAM_ABBREVIATION"].drop_duplicates().head(2).values
        )
        if len(abbrs) >= 2:
            home_abbr, away_abbr = abbrs[0], abbrs[1]
        else:
            home_abbr, away_abbr = "BOS", "LAL"
    else:
        home_abbr, away_abbr = "BOS", "LAL"

    print("--- Example prediction ---")
    try:
        prob = predict_matchup(home_abbr, away_abbr, as_of_date=None)
        print(f"P(home win | {home_abbr} vs {away_abbr}): {prob:.4f}")
    except Exception as e:
        print(f"Example prediction failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
