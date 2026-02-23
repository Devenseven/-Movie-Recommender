"""
app.py
------
Streamlit web UI for the Movie Recommendation System.

Features:
  - Dropdown to select a User ID
  - "Recommend for User" → top-N movies with predicted ratings
  - Text input to search for similar movies (handles typos)
  - RMSE score displayed on the evaluation tab
  - Model loaded only ONCE per session via @st.cache_resource
"""

import os
import sys

# Ensure `src/` is importable when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from src.data_loader import (
    load_ratings,
    load_movies,
    merge_data,
)
from src.model_builder import get_traintest_split, train_svd_model
from src.recommend import get_recommendations_for_user, get_similar_movies
from src.evaluate import evaluate_model

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global background ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(110, 72, 255, 0.35);
    }

    /* ── Rank badge ── */
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6e48ff, #c44dff);
        color: white;
        border-radius: 50%;
        width: 2rem; height: 2rem;
        line-height: 2rem;
        text-align: center;
        font-weight: 700;
        margin-right: 0.8rem;
    }

    /* ── Stars bar ── */
    .stars-bar-bg {
        background: rgba(255,255,255,0.12);
        border-radius: 20px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .stars-bar-fill {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        border-radius: 20px;
        height: 8px;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #c44dff;
        margin-bottom: 1rem;
        letter-spacing: 0.04em;
    }

    /* ── RMSE pill ── */
    .rmse-pill {
        display: inline-block;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: #000;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6e48ff, #c44dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.8rem !important;
        transition: box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(196,77,255,0.6) !important;
    }

    /* ── Hide Streamlit default footer ── */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Data & model loading (cached — runs once)
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_resource(show_spinner="🎬 Loading data and training model — this takes ~30s on first run...")
def load_everything():
    """
    Load data, train SVD, and evaluate.
    Cached so this only runs ONCE per Streamlit session.

    Returns a dict with:
      ratings_df, movies_df, algo, test_df, rmse
    """
    ratings = load_ratings(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = load_movies(os.path.join(DATA_DIR, "movies.csv"))

    train_df, test_df = get_traintest_split(ratings, test_size=0.2)
    algo = train_svd_model(train_df, n_factors=100, n_epochs=20)
    rmse = evaluate_model(algo, test_df)

    return {
        "ratings_df": ratings,
        "movies_df": movies,
        "algo": algo,
        "test_df": test_df,
        "rmse": rmse,
    }


# ─────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────

state = load_everything()
ratings_df: pd.DataFrame = state["ratings_df"]
movies_df:  pd.DataFrame = state["movies_df"]
algo                      = state["algo"]
rmse: float               = state["rmse"]


# ─────────────────────────────────────────────
# Sidebar — global settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("---")
    st.markdown("**Powered by SVD Collaborative Filtering**")
    st.markdown(
        f"📊 **{ratings_df['userId'].nunique():,}** users · "
        f"**{ratings_df['movieId'].nunique():,}** movies · "
        f"**{len(ratings_df):,}** ratings"
    )
    st.markdown("---")
    top_n = st.slider("🔢 Number of recommendations", min_value=3, max_value=20, value=10)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center'>Model RMSE &nbsp;"
        f"<span class='rmse-pill'>{rmse:.4f}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#999'>Lower is better · typical SVD range: 0.87–0.93</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Main area — two tabs
# ─────────────────────────────────────────────

tab_user, tab_similar, tab_eval = st.tabs([
    "👤 User Recommendations",
    "🎭 Similar Movies",
    "📊 Model Evaluation",
])


# ── TAB 1: User Recommendations ──────────────────────────────────────────────

with tab_user:
    st.markdown("<div class='section-header'>👤 Top Movie Picks — Just For You</div>",
                unsafe_allow_html=True)

    all_users = sorted(ratings_df["userId"].unique())
    selected_user = st.selectbox(
        "Select a User ID",
        options=all_users,
        format_func=lambda uid: f"User {uid}",
        key="user_select",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        rec_clicked = st.button("🚀 Recommend for User", key="rec_btn")

    if rec_clicked:
        with st.spinner(f"Finding top {top_n} picks for User {selected_user}…"):
            try:
                recs = get_recommendations_for_user(
                    user_id=selected_user,
                    algo=algo,
                    movies_df=movies_df,
                    ratings_df=ratings_df,
                    n=top_n,
                )
                st.success(f"✅ Top {len(recs)} recommendations for **User {selected_user}**")

                for rank, (title, pred_rating) in enumerate(recs, start=1):
                    bar_pct = int((pred_rating / 5.0) * 100)
                    st.markdown(
                        f"""
                        <div class='card'>
                          <span class='rank-badge'>{rank}</span>
                          <strong style='font-size:1.05rem'>{title}</strong>
                          <br>
                          <span style='color:#ffd200;font-size:0.9rem'>
                            Predicted rating: <b>{pred_rating}</b> / 5.0
                          </span>
                          <div class='stars-bar-bg'>
                            <div class='stars-bar-fill' style='width:{bar_pct}%'></div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except ValueError as e:
                st.error(str(e))


# ── TAB 2: Similar Movies ─────────────────────────────────────────────────────

with tab_similar:
    st.markdown("<div class='section-header'>🎭 Find Similar Movies</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Enter a movie title (typos are OK — we'll find the closest match).",
        unsafe_allow_html=False,
    )

    movie_query = st.text_input(
        "Movie title",
        placeholder="e.g. Toy Story, Schindler's List, The Godfather…",
        key="movie_input",
    )

    col_btn2, _ = st.columns([1, 4])
    with col_btn2:
        sim_clicked = st.button("🔍 Find Similar Movies", key="sim_btn")

    if sim_clicked and movie_query.strip():
        with st.spinner(f"Searching for movies similar to '{movie_query}'…"):
            try:
                results, matched_title = get_similar_movies(
                    movie_title=movie_query.strip(),
                    algo=algo,
                    movies_df=movies_df,
                    ratings_df=ratings_df,
                    n=top_n,
                )

                if matched_title.lower() != movie_query.strip().lower():
                    st.info(f"🔤 Matched your input to: **{matched_title}**")

                st.success(f"✅ Top {len(results)} movies similar to **{matched_title}**")

                for rank, (title, sim_score) in enumerate(results, start=1):
                    bar_pct = int(sim_score * 100)
                    st.markdown(
                        f"""
                        <div class='card'>
                          <span class='rank-badge'>{rank}</span>
                          <strong style='font-size:1.05rem'>{title}</strong>
                          <br>
                          <span style='color:#38ef7d;font-size:0.9rem'>
                            Cosine similarity: <b>{sim_score:.4f}</b>
                          </span>
                          <div class='stars-bar-bg'>
                            <div class='stars-bar-fill'
                              style='width:{max(bar_pct,4)}%;
                                     background:linear-gradient(90deg,#11998e,#38ef7d)'>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except ValueError as e:
                st.error(str(e))
    elif sim_clicked:
        st.warning("Please enter a movie title first.")


# ── TAB 3: Evaluation ────────────────────────────────────────────────────────

with tab_eval:
    st.markdown("<div class='section-header'>📊 Model Evaluation — RMSE</div>",
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Algorithm", "SVD")
    col_b.metric("Test Split", "20%")
    col_c.metric("RMSE", f"{rmse:.4f}")

    st.markdown("---")
    st.markdown(
        """
        ### What is RMSE?

        **Root Mean Squared Error** measures how far the model's predicted ratings
        are from the actual ratings in the test set:

        $$RMSE = \\sqrt{\\frac{1}{N} \\sum_{(u,i) \\in \\text{test}} (r_{ui} - \\hat{r}_{ui})^2}$$

        | Term | Meaning |
        |------|---------|
        | $r_{ui}$ | True rating by user $u$ on item $i$ |
        | $\\hat{r}_{ui}$ | Predicted rating from SVD |
        | $N$ | Number of test-set ratings |

        A lower RMSE indicates better accuracy. SVD on MovieLens 100k typically achieves
        **0.87–0.93**, which is excellent for collaborative filtering.
        """,
        unsafe_allow_html=False,
    )

    st.markdown("---")
    st.markdown("### How SVD Works")
    st.markdown(
        """
        SVD factorizes the sparse user-item rating matrix $R$ into:

        $$R \\approx U \\cdot \\Sigma \\cdot V^T$$

        During training, the model learns:
        - **$p_u$** — a latent vector for each user (their "taste profile")
        - **$q_i$** — a latent vector for each item (its "character profile")
        - **$b_u, b_i$** — bias terms for users and items

        The predicted rating is:

        $$\\hat{r}_{ui} = \\mu + b_u + b_i + q_i \\cdot p_u^T$$

        These vectors are learned by minimizing the regularized squared error
        using Stochastic Gradient Descent (SGD).
        """
    )
