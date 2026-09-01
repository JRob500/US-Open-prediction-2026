import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parents[1]))
from features.build_features import build_current_player_states  # noqa: E402
from models.perceptron import Perceptron  # noqa: E402 (needed for joblib.load to resolve the class)
from api.schemas import PredictRequest, PredictResponse, PlayersResponse  # noqa: E402
 
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
 
DEFAULT_SURFACE = "Hard"          # US Open is a hard-court event
DEFAULT_TOURNEY_LEVEL = "G"       # Grand Slam
BEST_OF_BY_TOUR = {"atp": 5, "wta": 3}
 
app = FastAPI(title="US Open Match Predictor")
 
# Populated at startup -- see _load_tour() below
STATE: dict[str, dict] = {}
 
 
def _name_lookup(static_info: dict) -> dict[str, str]:
    """Maps a lowercased player name -> player_id, for case-insensitive
    lookup by name based on what a person actually types """
    lookup = {}
    for player_id, info in static_info.items():
        lookup[str(info["name"]).strip().lower()] = player_id
    return lookup
 
 
def _load_tour(tour: str) -> dict:
    print(f"Loading {tour.upper()} model and match history...")
    pipeline = joblib.load(MODEL_DIR / f"{tour}_pipeline.pkl")
 
    matches_path = DATA_DIR / f"{tour}_matches_clean.csv"
    df = pd.read_csv(matches_path, parse_dates=["tourney_date"], low_memory=False)
 
    players, h2h, static_info = build_current_player_states(df)
    name_to_id = _name_lookup(static_info)
 
    print(f"  {len(static_info)} players known for {tour.upper()}")
    return {
        "pipeline": pipeline,
        "players": players,
        "h2h": h2h,
        "static_info": static_info,
        "name_to_id": name_to_id,
    }
 
def startup():
    for tour in ["atp", "wta"]:
        try:
            STATE[tour] = _load_tour(tour)
        except FileNotFoundError as e:
            print(f"WARNING: could not load {tour.upper()} data/model ({e}). "
                  f"/predict and players for '{tour}' will fail until this is fixed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield

app = FastAPI(title="US Open Match Predictor", lifespan=lifespan) 
 
def _resolve_player(tour: str, name: str) -> str:
    name_to_id = STATE[tour]["name_to_id"]
    player_id = name_to_id.get(name.strip().lower())
    if player_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{name}' not found for tour '{tour}'. "
                   f"Check GET /players?tour={tour} for valid names.",
        )
    return player_id
 
 
def _build_feature_row(tour: str, a_id: str, b_id: str, round_: str) -> pd.DataFrame:
    tour_state = STATE[tour]
    players = tour_state["players"]
    h2h = tour_state["h2h"]
    static_info = tour_state["static_info"]
 
    a_player = players[a_id]
    b_player = players[b_id]
    a_info = static_info[a_id]
    b_info = static_info[b_id]
 
    now = pd.Timestamp.now().normalize()
    a_snap = a_player.snapshot(DEFAULT_SURFACE, now)
    b_snap = b_player.snapshot(DEFAULT_SURFACE, now)
    h2h_stats = h2h.stats(a_id, b_id)
 
    row = {
        "elo_diff": a_snap["elo"] - b_snap["elo"],
        "surface_elo_diff": a_snap["surface_elo"] - b_snap["surface_elo"],
        "rank_diff": (b_info["rank"] or 0) - (a_info["rank"] or 0),
        "rank_points_diff": (a_info["rank_points"] or 0) - (b_info["rank_points"] or 0),
        "win_pct_last10_diff": a_snap["win_pct_last10"] - b_snap["win_pct_last10"],
        "surface_win_pct_last10_diff": (
            a_snap["surface_win_pct_last10"] - b_snap["surface_win_pct_last10"]
        ),
        "age_diff": (a_info["age"] or 0) - (b_info["age"] or 0),
        "ht_diff": (a_info["ht"] or 0) - (b_info["ht"] or 0),
        "days_since_last_match_diff": (
            a_snap["days_since_last_match"] - b_snap["days_since_last_match"]
        ),
        "h2h_matches": h2h_stats["h2h_matches"],
        "h2h_player_a_win_pct": h2h_stats["h2h_player_a_win_pct"],
        "player_a_matches_played": a_snap["matches_played"],
        "player_b_matches_played": b_snap["matches_played"],
        "best_of": BEST_OF_BY_TOUR[tour],
        "player_a_unranked": int(bool(a_info["unranked"])),
        "player_b_unranked": int(bool(b_info["unranked"])),
        "player_a_returning_from_layoff": int(a_snap["returning_from_layoff"]),
        "player_b_returning_from_layoff": int(b_snap["returning_from_layoff"]),
        "surface": DEFAULT_SURFACE,
        "tourney_level": DEFAULT_TOURNEY_LEVEL,
        "round": round_,
    }
    return pd.DataFrame([row]), a_snap, b_snap
 
 
@app.get("/players", response_model=PlayersResponse)
def get_players(tour: str = Query(..., pattern="^(atp|wta)$")):
    if tour not in STATE:
        raise HTTPException(status_code=503, detail=f"'{tour}' data not loaded")
    names = sorted(info["name"] for info in STATE[tour]["static_info"].values())
    return PlayersResponse(tour=tour, players=names)
 
 
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    tour = request.tour.lower()
    if tour not in STATE:
        raise HTTPException(status_code=400, detail="tour must be 'atp' or 'wta'")
 
    a_id = _resolve_player(tour, request.player_a)
    b_id = _resolve_player(tour, request.player_b)
 
    X, a_snap, b_snap = _build_feature_row(tour, a_id, b_id, request.round)
    pipeline = STATE[tour]["pipeline"]
    proba = pipeline.predict_proba(X)[0]  # [P(player_b wins), P(player_a wins)]
 
    a_info = STATE[tour]["static_info"][a_id]
    b_info = STATE[tour]["static_info"][b_id]
 
    return PredictResponse(
        tour=tour,
        player_a=a_info["name"],
        player_b=b_info["name"],
        player_a_win_probability=round(float(proba[1]), 4),
        player_b_win_probability=round(float(proba[0]), 4),
        player_a_elo=round(a_snap["elo"], 1),
        player_b_elo=round(b_snap["elo"], 1),
        player_a_rank=a_info["rank"],
        player_b_rank=b_info["rank"],
    )
 
 
@app.get("/")
def root():
    return {
        "message": "US Open Match Predictor",
        "tours_loaded": list(STATE.keys()),
        "endpoints": ["/players?tour=atp|wta", "/predict"],
    }