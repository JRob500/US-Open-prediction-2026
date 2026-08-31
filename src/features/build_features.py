import argparse
import random
from collections import defaultdict, deque
from pathlib import Path
 
import numpy as np
import pandas as pd
 
DEFAULT_ELO = 1500.0
ELO_K = 32.0
FORM_WINDOW = 10          # matches, for rolling win %
LONG_LAYOFF_DAYS = 60     # threshold for "returning from injury" flag
 
# Elo Rating
 
def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula: probability A beats B
    given their current ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
 
 
def update_elo(rating_winner: float, rating_loser: float, k: float = ELO_K):
    """Returns (new_winner_rating, new_loser_rating) after one match."""
    expected_winner = expected_score(rating_winner, rating_loser)
    expected_loser = 1.0 - expected_winner
    new_winner = rating_winner + k * (1.0 - expected_winner)
    new_loser = rating_loser + k * (0.0 - expected_loser)
    return new_winner, new_loser
 
 
# Per-player running state

class PlayerState:
    """Holds everything we need to know about a player's history AS OF
    a given point in time, updated incrementally match by match."""
 
    def __init__(self):
        self.elo = DEFAULT_ELO
        self.surface_elo = defaultdict(lambda: DEFAULT_ELO)
        self.recent_results = deque(maxlen=FORM_WINDOW)          # 1=win, 0=loss
        self.recent_surface_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
        self.matches_played = 0
        self.last_match_date = None
 
    def win_pct(self, results: deque) -> float:
        if len(results) == 0:
            return 0.5  # neutral prior for a player with no history yet
        return sum(results) / len(results)
 
    def rolling_win_pct(self) -> float:
        return self.win_pct(self.recent_results)
 
    def rolling_surface_win_pct(self, surface: str) -> float:
        return self.win_pct(self.recent_surface_results[surface])
 
    def days_since_last_match(self, current_date) -> float:
        if self.last_match_date is None:
            return 365.0  # neutral "fully rested" default for a debut
        return (current_date - self.last_match_date).days
 
    def snapshot(self, surface: str, current_date) -> dict:
        """Features as of RIGHT NOW, before today's match is applied."""
        return {
            "elo": self.elo,
            "surface_elo": self.surface_elo[surface],
            "win_pct_last10": self.rolling_win_pct(),
            "surface_win_pct_last10": self.rolling_surface_win_pct(surface),
            "matches_played": self.matches_played,
            "days_since_last_match": self.days_since_last_match(current_date),
            "returning_from_layoff": self.days_since_last_match(current_date) > LONG_LAYOFF_DAYS and self.matches_played > 0,
        }
 
    def apply_result(self, won: bool, surface: str, match_date) -> None:
        self.recent_results.append(1 if won else 0)
        self.recent_surface_results[surface].append(1 if won else 0)
        self.matches_played += 1
        self.last_match_date = match_date
 
 
class HeadToHead:
    """Tracks win counts between every pair of players seen so far."""
 
    def __init__(self):
        self._wins = defaultdict(int)  # (winner_id, loser_id) -> count
 
    def record(self, key_a: str, key_b: str) -> None:
        self._wins[(key_a, key_b)] += 1
 
    def stats(self, player_a_id: str, player_b_id: str) -> dict:
        a_wins = self._wins[(player_a_id, player_b_id)]
        b_wins = self._wins[(player_b_id, player_a_id)]
        total = a_wins + b_wins
        win_pct = a_wins / total if total > 0 else 0.5
        return {"h2h_matches": total, "h2h_player_a_win_pct": win_pct}

def build_current_player_states(df: pd.DataFrame):
   
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
 
    players: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h = HeadToHead()
    static_info: dict[str, dict] = {}
 
    for _, match in df.iterrows():
        winner_id = match["winner_id"]
        loser_id = match["loser_id"]
        surface = match.get("surface", "Unknown")
        match_date = match["tourney_date"]
 
        winner_state = players[winner_id]
        loser_state = players[loser_id]
 
        new_winner_elo, new_loser_elo = update_elo(winner_state.elo, loser_state.elo)
        winner_state.elo = new_winner_elo
        loser_state.elo = new_loser_elo
 
        new_winner_surf_elo, new_loser_surf_elo = update_elo(
            winner_state.surface_elo[surface], loser_state.surface_elo[surface]
        )
        winner_state.surface_elo[surface] = new_winner_surf_elo
        loser_state.surface_elo[surface] = new_loser_surf_elo
 
        winner_state.apply_result(won=True, surface=surface, match_date=match_date)
        loser_state.apply_result(won=False, surface=surface, match_date=match_date)
        h2h.record(winner_id, loser_id)
 
        # Always overwrite with the latest -- since rows are processed in
        # chronological order, whatever is written last IS the most recent.
        static_info[winner_id] = {
            "name": match["winner_name"], "rank": match["winner_rank"],
            "rank_points": match["winner_rank_points"], "age": match["winner_age"],
            "ht": match["winner_ht"], "hand": match["winner_hand"],
            "unranked": match["winner_unranked"],
        }
        static_info[loser_id] = {
            "name": match["loser_name"], "rank": match["loser_rank"],
            "rank_points": match["loser_rank_points"], "age": match["loser_age"],
            "ht": match["loser_ht"], "hand": match["loser_hand"],
            "unranked": match["loser_unranked"],
        }
 
    return players, h2h, static_info

# Main feature-building loop
 
def build_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    rng = random.Random(seed)
 
    players: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h = HeadToHead()
 
    rows = []
 
    for _, match in df.iterrows():
        winner_id = match["winner_id"]
        loser_id = match["loser_id"]
        surface = match.get("surface", "Unknown")
        match_date = match["tourney_date"]
 
        winner_state = players[winner_id]
        loser_state = players[loser_id]
 
        # Reads the features before this match affects anything 
        winner_snapshot = winner_state.snapshot(surface, match_date)
        loser_snapshot = loser_state.snapshot(surface, match_date)
        h2h_stats = h2h.stats(winner_id, loser_id)  # from winner's perspective
 
        # Winner could become either player_a or player_b 
        winner_is_a = rng.random() < 0.5
 
        if winner_is_a:
            a_id, b_id = winner_id, loser_id
            a_name, b_name = match["winner_name"], match["loser_name"]
            a_snap, b_snap = winner_snapshot, loser_snapshot
            a_rank, b_rank = match["winner_rank"], match["loser_rank"]
            a_rank_pts, b_rank_pts = match["winner_rank_points"], match["loser_rank_points"]
            a_age, b_age = match["winner_age"], match["loser_age"]
            a_ht, b_ht = match["winner_ht"], match["loser_ht"]
            a_hand, b_hand = match["winner_hand"], match["loser_hand"]
            a_unranked, b_unranked = match["winner_unranked"], match["loser_unranked"]
            h2h_a_win_pct = h2h_stats["h2h_player_a_win_pct"]
            target = 1
        else:
            a_id, b_id = loser_id, winner_id
            a_name, b_name = match["loser_name"], match["winner_name"]
            a_snap, b_snap = loser_snapshot, winner_snapshot
            a_rank, b_rank = match["loser_rank"], match["winner_rank"]
            a_rank_pts, b_rank_pts = match["loser_rank_points"], match["winner_rank_points"]
            a_age, b_age = match["loser_age"], match["winner_age"]
            a_ht, b_ht = match["loser_ht"], match["winner_ht"]
            a_hand, b_hand = match["loser_hand"], match["winner_hand"]
            a_unranked, b_unranked = match["loser_unranked"], match["winner_unranked"]
            h2h_a_win_pct = 1.0 - h2h_stats["h2h_player_a_win_pct"] if h2h_stats["h2h_matches"] > 0 else 0.5
            target = 0
 
        rows.append({
            "tourney_id": match["tourney_id"],
            "tourney_date": match_date,
            "surface": surface,
            "best_of": match.get("best_of"),
            "round": match.get("round"),
            "tourney_level": match.get("tourney_level"),
 
            "player_a_id": a_id,
            "player_a_name": a_name,
            "player_a_rank": a_rank,
            "player_a_rank_points": a_rank_pts,
            "player_a_age": a_age,
            "player_a_ht": a_ht,
            "player_a_hand": a_hand,
            "player_a_unranked": a_unranked,
            "player_a_elo": a_snap["elo"],
            "player_a_surface_elo": a_snap["surface_elo"],
            "player_a_win_pct_last10": a_snap["win_pct_last10"],
            "player_a_surface_win_pct_last10": a_snap["surface_win_pct_last10"],
            "player_a_matches_played": a_snap["matches_played"],
            "player_a_days_since_last_match": a_snap["days_since_last_match"],
            "player_a_returning_from_layoff": a_snap["returning_from_layoff"],
 
            "player_b_id": b_id,
            "player_b_name": b_name,
            "player_b_rank": b_rank,
            "player_b_rank_points": b_rank_pts,
            "player_b_age": b_age,
            "player_b_ht": b_ht,
            "player_b_hand": b_hand,
            "player_b_unranked": b_unranked,
            "player_b_elo": b_snap["elo"],
            "player_b_surface_elo": b_snap["surface_elo"],
            "player_b_win_pct_last10": b_snap["win_pct_last10"],
            "player_b_surface_win_pct_last10": b_snap["surface_win_pct_last10"],
            "player_b_matches_played": b_snap["matches_played"],
            "player_b_days_since_last_match": b_snap["days_since_last_match"],
            "player_b_returning_from_layoff": b_snap["returning_from_layoff"],
 
            "h2h_matches": h2h_stats["h2h_matches"],
            "h2h_player_a_win_pct": h2h_a_win_pct,
 
            "player_a_wins": target,
        })
 
        # Updates state after reading, using the real result 
        new_winner_elo, new_loser_elo = update_elo(winner_state.elo, loser_state.elo)
        winner_state.elo = new_winner_elo
        loser_state.elo = new_loser_elo
 
        new_winner_surf_elo, new_loser_surf_elo = update_elo(
            winner_state.surface_elo[surface], loser_state.surface_elo[surface]
        )
        winner_state.surface_elo[surface] = new_winner_surf_elo
        loser_state.surface_elo[surface] = new_loser_surf_elo
 
        winner_state.apply_result(won=True, surface=surface, match_date=match_date)
        loser_state.apply_result(won=False, surface=surface, match_date=match_date)
        h2h.record(winner_id, loser_id)
 
    features = pd.DataFrame(rows)
    return features
 
 
def main():
    parser = argparse.ArgumentParser(description="Phase 2 feature engineering")
    parser.add_argument("--input", required=True, help="Cleaned archive CSV from clean_data.py")
    parser.add_argument("--output", required=True, help="Output features CSV path")
    parser.add_argument("--tour", choices=["atp", "wta"], required=True, help="Just for logging/clarity, doesn't change logic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the player_a/player_b coin flip")
    args = parser.parse_args()
 
    print(f"Loading {args.input} ({args.tour.upper()})...")
    df = pd.read_csv(args.input, low_memory=False, parse_dates=["tourney_date"])
    print(f"Loaded {len(df)} matches")
 
    features = build_features(df, seed=args.seed)
 
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
 
    print(f"\nBuilt {len(features)} feature rows")
    print(f"player_a_wins balance: {features['player_a_wins'].mean():.3f} "
          f"(should be close to 0.5)")
    print(f"Saved to {out_path}")
 
 
if __name__ == "__main__":
    main()