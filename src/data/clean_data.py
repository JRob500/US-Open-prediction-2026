import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_IDS = ["tourney_id", "match_num", "winner_id", "loser_id", "score"]
SERVING_STATS = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced","l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]

PLAYER_HEIGHT = ["winner_ht", "loser_ht"]
PLAYER_AGE = ["winner_age", "loser_age"]

MIN_HEIGHT = 140
MAX_HEIGHT = 220

MIN_AGE = 13
MAX_AGE = 50

MIN_RANK = 1


def log_step(label: str, before: int, after: int) -> None:
    removed = before - after
    print(f"[{label}] removed {removed:>6} rows  ({before} -> {after})")


def load_raw(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory = False)
        df["source_file"] = Path(p).name
        frames.append(df)
        print(f"Loaded {p}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index = True)
    combined["tourney_date"] = pd.to_datetime(combined["tourney_date"], format = "%Y%m%d", errors = "coerce")
    print(f"\nCombined raw dataset: {len(combined)} rows\n")
    return combined


def drop_missing_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset = COLUMN_IDS)
    log_step("drop_missing_required_fields", before, len(df))
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset = ["tourney_id", "match_num"], keep ="first")
    log_step("drop_duplicates", before, len(df))
    return df


def drop_walkovers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    score = df["score"].astype(str)
    is_walkover = score.str.contains("W/O", case = False, na = False) | (score.str.strip() == "")
    df = df.loc[~is_walkover].copy()
    log_step("drop_walkovers", before, len(df))
    return df


def drop_bad_physical_values(df: pd.DataFrame) -> pd.DataFrame:
    before_rows = len(df)

    for col in PLAYER_HEIGHT:
        bad = ~df[col].between(MIN_HEIGHT, MAX_HEIGHT) & df[col].notna()
        n_bad = bad.sum()
        if n_bad:
            print(f"  -> {col}: {n_bad} implausible values set to NaN")
        df.loc[bad, col] = np.nan

    for col in PLAYER_AGE:
        bad = ~df[col].between(MIN_AGE, MAX_AGE) & df[col].notna()
        n_bad = bad.sum()
        if n_bad:
            print(f"  -> {col}: {n_bad} implausible values set to NaN")
        df.loc[bad, col] = np.nan

    for col in ["winner_rank", "loser_rank"]:
        bad = (df[col] < MIN_RANK) & df[col].notna()
        n_bad = bad.sum()
        if n_bad:
            print(f"  -> {col}: {n_bad} implausible values set to NaN")
        df.loc[bad, col] = np.nan

    log_step("drop_bad_physical_values (rows unchanged, bad cells -> NaN)", before_rows, len(df))
    return df


def filter_singles_only(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    is_doubles_name = (
        df["winner_name"].astype(str).str.contains("/", na = False)
        | df["loser_name"].astype(str).str.contains("/", na = False)
    )
    df = df.loc[~is_doubles_name].copy()
    log_step("filter_singles_only", before, len(df))
    return df


def filter_year_range(df: pd.DataFrame, min_year: int | None) -> pd.DataFrame:
    if min_year is None:
        return df
    before = len(df)
    df = df.loc[df["tourney_date"].dt.year >= min_year].copy()
    log_step(f"filter_year_range (>= {min_year})", before, len(df))
    return df

def flag_retirements(df: pd.DataFrame) -> pd.DataFrame:
    score = df["score"].astype(str)
    df["retired"] = score.str.contains("RET", case = False, na = False)
    df["defaulted"] = score.str.contains("DEF", case = False, na = False)
    n = int(df["retired"].sum() + df["defaulted"].sum())
    print(f"[flag_retirements] flagged {n} retirement/default matches (kept, not dropped)")
    return df


def flag_and_impute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    Max_Rank = df[["winner_rank", "loser_rank"]].max().max()
    fallback_rank = int(Max_Rank) + 1 if pd.notna(Max_Rank) else 2000

    df["winner_unranked"] = df["winner_rank"].isna()
    df["loser_unranked"] = df["loser_rank"].isna()

    df["winner_rank"] = df["winner_rank"].fillna(fallback_rank)
    df["loser_rank"] = df["loser_rank"].fillna(fallback_rank)
    df["winner_rank_points"] = df["winner_rank_points"].fillna(0)
    df["loser_rank_points"] = df["loser_rank_points"].fillna(0)

    n_flagged = int(df["winner_unranked"].sum() + df["loser_unranked"].sum())
    print(f"[flag_and_impute_ranks] flagged {n_flagged} unranked player-appearances "
          f"(imputed rank = {fallback_rank}, rank_points = 0)")
    return df


def flag_missing_serve_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["has_serve_stats"] = df[SERVING_STATS].notna().all(axis = 1)
    n_missing = int((~df["has_serve_stats"]).sum())
    pct = 100 * n_missing / len(df) if len(df) else 0
    print(f"[flag_missing_serve_stats] {n_missing} rows ({pct:.1f}%) missing " f"at least one serve stat -- left as NaN, flagged via has_serve_stats")
    return df


def flag_handedness_and_indoor(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["winner_hand", "loser_hand"]:
        df[col] = df[col].where(df[col].isin(["R", "L"]), other = "U")
    return df


def split_completed_vs_upcoming(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = df["score"].astype(str).str.strip()
    has_score = score.notna() & (score != "") & (score.str.lower() != "nan")
    has_players = df["winner_id"].notna() & df["loser_id"].notna()

    is_completed = has_score & has_players
    completed = df.loc[is_completed].copy()
    upcoming = df.loc[~is_completed].copy()

    print(f"[split_completed_vs_upcoming] {len(completed)} completed matches, " f"{len(upcoming)} upcoming/scheduled fixtures")
    return completed, upcoming


def clean_archive(paths: list[str], min_year: int | None) -> pd.DataFrame:
    df = load_raw(paths)

    df = drop_missing_required_fields(df)
    df = drop_duplicates(df)
    df = drop_walkovers(df)
    df = filter_singles_only(df)
    df = filter_year_range(df, min_year)
    df = drop_bad_physical_values(df)

    df = flag_retirements(df)
    df = flag_and_impute_ranks(df)
    df = flag_missing_serve_stats(df)
    df = flag_handedness_and_indoor(df)

    df = df.sort_values("tourney_date").reset_index(drop = True)

    print(f"\nFinal cleaned dataset: {len(df)} rows, {df['tourney_date'].dt.year.min()}"
          f"-{df['tourney_date'].dt.year.max()}\n")
    return df


def clean_ongoing(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw(paths)

    df = drop_duplicates(df)
    df = filter_singles_only(df)

    completed, upcoming = split_completed_vs_upcoming(df)

    # Completed matches get the same real-match cleaning as archive rows
    if len(completed):
        completed = drop_walkovers(completed)
        completed = drop_bad_physical_values(completed)
        completed = flag_retirements(completed)
        completed = flag_and_impute_ranks(completed)
        completed = flag_missing_serve_stats(completed)
        completed = flag_handedness_and_indoor(completed)
        completed = completed.sort_values("tourney_date").reset_index(drop=True)

    # Upcoming fixtures: keep as-is, just tidy column order for readability.
    # No stat cleaning needed -- these matches haven't happened, so
    # w_ace/l_ace/etc. are meaningless (always NaN) and can be dropped.
    if len(upcoming):
        fixture_cols = [
            "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
            "indoor", "tourney_date", "match_num", "round", "best_of",
            "winner_seed", "winner_entry", "winner_name", "winner_rank",
            "loser_seed", "loser_entry", "loser_name", "loser_rank",
            "source_file",
        ]
        available_cols = [c for c in fixture_cols if c in upcoming.columns]
        upcoming = upcoming[available_cols].reset_index(drop=True)

    print(f"\nOngoing file cleaned: {len(completed)} completed, {len(upcoming)} upcoming\n")
    return completed, upcoming


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Cleaning for US Open match data")
    parser.add_argument("--mode", choices = ["archive", "ongoing"], default = "archive", help = "'archive' for frozen yearly ATP/WTA files, ""'ongoing' for live in-progress tournament files")
    parser.add_argument("--input", nargs = "+", required = True, help = "Raw CSV path(s) or glob pattern(s)")
    parser.add_argument("--output", required = True, help = "Archive mode: output CSV path. ""Ongoing mode: output path WITHOUT .csv extension ""-- writes <output>_completed.csv and <output>_upcoming.csv")
    parser.add_argument("--min-year", type = int, default = None, help = "Archive mode only: drop matches before this year")
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    if not paths:
        sys.exit("No input files found.")

    if args.mode == "archive":
        cleaned = clean_archive(paths, args.min_year)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        cleaned.to_csv(out_path, index = False)
        print(f"Saved cleaned dataset to {out_path}")

    else: 
        completed, upcoming = clean_ongoing(paths)
        out_base = Path(args.output)
        out_base.parent.mkdir(parents = True, exist_ok = True)

        completed_path = out_base.with_name(out_base.name + "_completed.csv")
        upcoming_path = out_base.with_name(out_base.name + "_upcoming.csv")

        completed.to_csv(completed_path, index = False)
        upcoming.to_csv(upcoming_path, index = False)
        print(f"Saved completed matches to {completed_path}")
        print(f"Saved upcoming fixtures to {upcoming_path}")


if __name__ == "__main__":
    main()