import random
 
import requests
 
ROUND_BY_PLAYERS_REMAINING = {
    128: "R128", 64: "R64", 32: "R32", 16: "R16",
    8: "QF", 4: "SF", 2: "F",
}
 
 
class ApiError(Exception):
    """Raised when the API is unreachable or returns an error we should
    surface clearly to the user, rather than letting a raw requests
    exception or a KeyError bubble up."""
    pass
 
 
def fetch_players(api_url: str, tour: str) -> list[str]:
    try:
        resp = requests.get(f"{api_url}/players", params={"tour": tour}, timeout=10)
    except requests.exceptions.ConnectionError:
        raise ApiError(
            f"Could not connect to the API at {api_url}. "
            f"Is it running? (uvicorn src.api.main:app --reload)"
        )
    if resp.status_code != 200:
        raise ApiError(f"Failed to fetch players ({resp.status_code}): {resp.text}")
    return resp.json()["players"]
 
 
def predict_match(api_url: str, tour: str, player_a: str, player_b: str,
                   round_: str = "R128") -> dict:
    try:
        resp = requests.post(
            f"{api_url}/predict",
            json={"tour": tour, "player_a": player_a, "player_b": player_b, "round": round_},
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Could not connect to the API at {api_url}.")
 
    if resp.status_code == 404:
        raise ApiError(resp.json().get("detail", "Player not found."))
    if resp.status_code == 422:
        raise ApiError(f"Invalid request: {resp.json().get('detail')}")
    if resp.status_code != 200:
        raise ApiError(f"Prediction failed ({resp.status_code}): {resp.text}")
 
    return resp.json()
 
 
def pick_winner(prediction: dict, mode: str, rng: random.Random) -> str:
    """mode is 'favorite' (always pick the higher-probability player) or
    'weighted' (draw a winner using the real probabilities as odds --
    upsets can happen, same as real tennis)."""
    p_a = prediction["player_a_win_probability"]
    if mode == "favorite":
        return prediction["player_a"] if p_a >= 0.5 else prediction["player_b"]
    else:  # weighted
        return prediction["player_a"] if rng.random() < p_a else prediction["player_b"]
 
 
def simulate_bracket(api_url: str, tour: str, players: list[str], mode: str = "favorite",
                      seed: int | None = None) -> list[dict]:
    """Simulates a full single-elimination bracket. `players` must be a
    power-of-2 list, in the seeded order they'll be paired (adjacent
    pairs each round). Returns a list of round dicts:
        [{"round_label": "R32", "matches": [ {prediction dict, winner}, ... ]}, ...]
    """
    n = len(players)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"Bracket size must be a power of 2 (2, 4, 8, 16, 32...), got {n}")
 
    rng = random.Random(seed)
    rounds_output = []
    current_round = list(players)
 
    while len(current_round) > 1:
        round_label = ROUND_BY_PLAYERS_REMAINING.get(len(current_round), "R128")
        matches = []
        next_round = []
 
        for i in range(0, len(current_round), 2):
            player_a, player_b = current_round[i], current_round[i + 1]
            prediction = predict_match(api_url, tour, player_a, player_b, round_label)
            winner = pick_winner(prediction, mode, rng)
            matches.append({"prediction": prediction, "winner": winner})
            next_round.append(winner)
 
        rounds_output.append({"round_label": round_label, "matches": matches})
        current_round = next_round
 
    return rounds_output
 
 
if __name__ == "__main__":
    API_URL = "http://127.0.0.1:8000"
    TOUR = "atp"
 
    print("Fetching players...")
    players = fetch_players(API_URL, TOUR)
    print(f"Got {len(players)} players\n")
 
    print("Testing single prediction...")
    pred = predict_match(API_URL, TOUR, players[0], players[1], "R32")
    print(pred, "\n")
 
    print("Testing error handling (bad player name)...")
    try:
        predict_match(API_URL, TOUR, "Not A Real Player", players[1])
    except ApiError as e:
        print(f"  Correctly raised ApiError: {e}\n")
 
    print("Simulating an 8-player bracket (favorite mode)...")
    import random as _r
    sample = _r.sample(players, 8)
    print("Seed players:", sample)
    bracket = simulate_bracket(API_URL, TOUR, sample, mode="favorite")
    for rnd in bracket:
        print(f"\n{rnd['round_label']}:")
        for m in rnd["matches"]:
            p = m["prediction"]
            print(f"  {p['player_a']} ({p['player_a_win_probability']:.0%}) vs "
                  f"{p['player_b']} ({p['player_b_win_probability']:.0%}) "
                  f"-> {m['winner']}")
    champion = bracket[-1]["matches"][0]["winner"]
    print(f"\nChampion: {champion}")