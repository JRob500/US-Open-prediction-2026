
# Two features:
#  - Single Match Predictor: pick two players, see a win probability
#  - Bracket Simulator: seed a power-of-2 bracket from real players,
#    simulate it round by round using live predictions from the API
 
# Run with:
#    streamlit run app/streamlit_app.py
 
#Requires the FastAPI service to be running separately:
#    uvicorn src.api.main:app --reload

 
import streamlit as st
 
from bracket_logic import ApiError, fetch_players, predict_match, simulate_bracket
 
st.set_page_config(page_title="US Open Match Predictor", layout="wide")
 
VALID_ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
BRACKET_SIZES = [4, 8, 16, 32]
 
 
# Sidebar -- shared config across both tabs
 
st.sidebar.title("US Open Match Predictor")
api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
tour = st.sidebar.radio("Tour", ["atp", "wta"], format_func=lambda t: "Men's (ATP)" if t == "atp" else "Women's (WTA)")
 
st.sidebar.markdown("---")
st.sidebar.caption(
    "This app calls a separately-running FastAPI service. "
    "If predictions fail, make sure it's running: "
    "`uvicorn src.api.main:app --reload`"
)
 
 
@st.cache_data(ttl=300)
def cached_players(api_url: str, tour: str) -> list[str]:
    return fetch_players(api_url, tour)
 
 
try:
    players = cached_players(api_url, tour)
except ApiError as e:
    st.error(f" error 404 {e}")
    st.stop()
 
if not players:
    st.warning(f"No players found for {tour.upper()}. Check that the API loaded this tour's data correctly.")
    st.stop()
 
 
# Tabs
 
tab_single, tab_bracket = st.tabs([" Single Match Predictor", "Bracket Simulator"])
 
with tab_single:
    st.header("Predict a single match")
 
    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Player A", players, index=0, key="single_a")
    with col2:
        # Default to a different player than A, if possible
        default_b_index = 1 if len(players) > 1 else 0
        player_b = st.selectbox("Player B", players, index=default_b_index, key="single_b")
 
    round_choice = st.selectbox("Round", VALID_ROUNDS, index=0)
 
    if player_a == player_b:
        st.warning("Player A and Player B are the same -- pick two different players for a meaningful prediction.")
 
    if st.button("Predict", type="primary", disabled=(player_a == player_b)):
        try:
            with st.spinner("Getting prediction..."):
                result = predict_match(api_url, tour, player_a, player_b, round_choice)
        except ApiError as e:
            st.error(f"{e}")
        else:
            st.subheader("Result")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(result["player_a"], f"{result['player_a_win_probability']:.0%}")
                st.progress(result["player_a_win_probability"])
                st.caption(f"Elo: {result['player_a_elo']:.0f} · Rank: {result['player_a_rank']:.0f}"
                           if result["player_a_rank"] else f"Elo: {result['player_a_elo']:.0f} · Unranked")
            with c2:
                st.metric(result["player_b"], f"{result['player_b_win_probability']:.0%}")
                st.progress(result["player_b_win_probability"])
                st.caption(f"Elo: {result['player_b_elo']:.0f} · Rank: {result['player_b_rank']:.0f}"
                           if result["player_b_rank"] else f"Elo: {result['player_b_elo']:.0f} · Unranked")
 
 
with tab_bracket:
    st.header("Simulate a bracket")
 
    col1, col2 = st.columns([1, 1])
    with col1:
        bracket_size = st.selectbox("Bracket size", BRACKET_SIZES, index=1)
    with col2:
        mode_label = st.radio(
            "Winner selection",
            ["Favorite always wins", "Weighted random (realistic upsets)"],
            horizontal=False,
        )
        mode = "favorite" if mode_label.startswith("Favorite") else "weighted"
 
    seed_players = st.multiselect(
        f"Pick exactly {bracket_size} players to seed the bracket",
        players,
        max_selections=bracket_size,
    )
 
    ready = len(seed_players) == bracket_size
 
    if not ready:
        st.info(f"Select {bracket_size - len(seed_players)} more player(s) to run the simulation.")
 
    if st.button("Simulate Bracket", type="primary", disabled=not ready):
        try:
            with st.spinner("Simulating..."):
                bracket = simulate_bracket(api_url, tour, seed_players, mode=mode)
        except ApiError as e:
            st.error(f"{e}")
        else:
            for rnd in bracket:
                st.subheader(rnd["round_label"])
                for m in rnd["matches"]:
                    p = m["prediction"]
                    winner = m["winner"]
                    a_won = winner == p["player_a"]
                    c1, c2, c3 = st.columns([3, 1, 3])
                    with c1:
                        label = f"**{p['player_a']}**" if a_won else p["player_a"]
                        st.markdown(label)
                        st.progress(p["player_a_win_probability"])
                        st.caption(f"{p['player_a_win_probability']:.0%}")
                    with c2:
                        st.markdown("<div style='text-align:center; padding-top: 20px;'>vs</div>",
                                    unsafe_allow_html=True)
                    with c3:
                        label = f"**{p['player_b']}**" if not a_won else p["player_b"]
                        st.markdown(label)
                        st.progress(p["player_b_win_probability"])
                        st.caption(f"{p['player_b_win_probability']:.0%}")
 
            champion = bracket[-1]["matches"][0]["winner"]
            st.success(f"## Champion: {champion}")