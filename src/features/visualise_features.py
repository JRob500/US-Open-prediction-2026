import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["player_a_elo"] - df["player_b_elo"]
    df["surface_elo_diff"] = df["player_a_surface_elo"] - df["player_b_surface_elo"]
    df["rank_diff"] = df["player_b_rank"] - df["player_a_rank"]
    df["rank_points_diff"] = df["player_a_rank_points"] - df["player_b_rank_points"]
    df["win_pct_last10_diff"] = df["player_a_win_pct_last10"] - df["player_b_win_pct_last10"]
    df["surface_win_pct_last10_diff"] = (
        df["player_a_surface_win_pct_last10"] - df["player_b_surface_win_pct_last10"]
    )
    df["age_diff"] = df["player_a_age"] - df["player_b_age"]
    df["ht_diff"] = df["player_a_ht"] - df["player_b_ht"]
    df["days_since_last_match_diff"] = (
        df["player_a_days_since_last_match"] - df["player_b_days_since_last_match"]
    )
    return df
 
 
NUMERIC_FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_diff", "rank_points_diff",
    "win_pct_last10_diff", "surface_win_pct_last10_diff",
    "age_diff", "ht_diff", "days_since_last_match_diff",
    "h2h_matches", "h2h_player_a_win_pct",
    "player_a_matches_played", "player_b_matches_played", "best_of",
]
TARGET = "player_a_wins"
 
 
def main():
    parser = argparse.ArgumentParser(description="Visualise feature correlations")
    parser.add_argument("--features", required=True)
    parser.add_argument("--tour", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
 
    df = pd.read_csv(args.features, low_memory=False)
    df = add_diff_features(df)
 
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr()
 
    # --- Full correlation heatmap (feature-vs-feature + target) ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"label": "Correlation"})
    plt.title(f"{args.tour.upper()} Feature Correlation Matrix")
    plt.tight_layout()
    out_path = Path(args.output_dir) / f"{args.tour}_feature_correlation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved full correlation heatmap to {out_path}")
 
    # --- Just "what correlates with the target", sorted -- often more
    # directly useful than the full matrix, since it answers "which
    # features actually matter for predicting a win" ---
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\nFeature correlation with player_a_wins (sorted by strength):")
    print(target_corr.to_string())
 
    plt.figure(figsize=(8, 6))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in target_corr.values]
    plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Correlation with player_a_wins")
    plt.title(f"{args.tour.upper()}: Which features predict a win?")
    plt.tight_layout()
    out_path2 = Path(args.output_dir) / f"{args.tour}_target_correlation.png"
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"Saved target-correlation bar chart to {out_path2}")
 
 
if __name__ == "__main__":
    main()