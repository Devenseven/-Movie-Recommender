"""
recommend.py
------------
Core recommendation logic:

  1. get_recommendations_for_user(user_id, algo, movies_df, ratings_df, n)
     - Finds all movies the user has NOT yet rated
     - Uses the trained SVD model to predict a rating for each unseen movie
     - Returns the top-n movies sorted by predicted rating (descending)

  2. get_similar_movies(movie_title, algo, movies_df, ratings_df, n)
     - Fuzzy-matches the input title against all known movie titles
     - Extracts the item latent vectors (q_i) from the SVD model
     - Computes cosine similarity between the target movie and all others
     - Returns the top-n most similar movies
"""

import difflib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


# ─────────────────────────────────────────────
# Helper: fuzzy title lookup
# ─────────────────────────────────────────────

def _fuzzy_match_title(query: str, all_titles: List[str]) -> str:
    """
    Return the closest matching movie title for a (potentially misspelled) query.

    Uses difflib.get_close_matches which implements the Ratcliff/Obershelp
    sequence matching algorithm.

    Raises:
        ValueError: if no close match is found at all
    """
    matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.4)
    if not matches:
        raise ValueError(
            f"No movie found matching '{query}'. "
            "Please check the title and try again."
        )
    return matches[0]


# ─────────────────────────────────────────────
# 1. User Recommendations
# ─────────────────────────────────────────────

def get_recommendations_for_user(
    user_id: int,
    algo,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Predict ratings for every movie the user hasn't seen and return the top-n.

    Args:
        user_id:    Target user ID (must exist in ratings_df)
        algo:       Trained SVDModel
        movies_df:  DataFrame with columns [movieId, title, genres]
        ratings_df: DataFrame with columns [userId, movieId, rating]
        n:          Number of recommendations to return

    Returns:
        List of (title, predicted_rating) tuples, sorted best-first.

    Raises:
        ValueError: if user_id is not found in the dataset.
    """
    all_user_ids = ratings_df["userId"].unique()
    if user_id not in all_user_ids:
        raise ValueError(
            f"User ID {user_id} not found in the dataset. "
            f"Valid IDs range from {all_user_ids.min()} to {all_user_ids.max()}."
        )

    # Movies the user has already rated
    rated_movie_ids = set(
        ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    )

    # Candidate movies: everything the user has NOT seen
    candidate_movies = movies_df[~movies_df["movieId"].isin(rated_movie_ids)]

    if candidate_movies.empty:
        raise ValueError(f"User {user_id} has rated every movie in the dataset!")

    # Predict rating for each candidate movie
    predictions: List[Tuple[str, float]] = []
    for _, row in candidate_movies.iterrows():
        est = algo.predict(user_id, row["movieId"])
        predictions.append((row["title"], round(est, 3)))

    # Sort by predicted rating (highest first) and return top-n
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


# ─────────────────────────────────────────────
# 2. Similar Movies
# ─────────────────────────────────────────────

def get_similar_movies(
    movie_title: str,
    algo,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n: int = 10,
) -> Tuple[List[Tuple[str, float]], str]:
    """
    Find the n most similar movies to a given title using latent item vectors.

    The SVD model stores a latent factor vector q_i for every item i.
    These vectors live in the same latent space, so cosine similarity
    between two q vectors measures how alike two movies are in that space.

    Args:
        movie_title: Movie title (typos handled via fuzzy matching)
        algo:        Trained SVDModel
        movies_df:   DataFrame with [movieId, title, genres]
        ratings_df:  DataFrame with [userId, movieId, rating]
        n:           Number of similar movies to return

    Returns:
        (results, matched_title) where results is a list of
        (title, similarity_score) tuples sorted best-first.

    Raises:
        ValueError: if the title cannot be matched or the model lacks item vectors.
    """
    # Only movies that appear in both the ratings data and the SVD trainset
    rated_movie_ids = set(ratings_df["movieId"].unique())
    known_movies = movies_df[movies_df["movieId"].isin(rated_movie_ids)].copy()
    known_movies = known_movies.reset_index(drop=True)

    all_titles = known_movies["title"].tolist()

    # Fuzzy-match the query title
    matched_title = _fuzzy_match_title(movie_title, all_titles)
    matched_row = known_movies[known_movies["title"] == matched_title].iloc[0]
    target_movie_id = matched_row["movieId"]

    # Map movieId → internal index
    trainset = algo.trainset  # returns self (SVDModel)
    try:
        target_inner_id = trainset.to_inner_iid(target_movie_id)
    except ValueError:
        raise ValueError(
            f"'{matched_title}' has no ratings in the training set and cannot "
            "be used for similarity lookup."
        )

    # Retrieve item latent factor matrix (n_items × n_factors)
    item_factors: np.ndarray = algo.qi  # shape: (n_items_in_trainset, n_factors)

    target_vector = item_factors[target_inner_id].reshape(1, -1)

    # Compute cosine similarity between target and ALL items
    similarities = cosine_similarity(target_vector, item_factors)[0]  # (n_items,)

    # Build list of (title, similarity) for all other movies
    results: List[Tuple[str, float]] = []
    for inner_id, sim_score in enumerate(similarities):
        if inner_id == target_inner_id:
            continue  # skip the movie itself
        try:
            raw_iid = int(trainset.to_raw_iid(inner_id))
        except Exception:
            continue
        title_rows = movies_df[movies_df["movieId"] == raw_iid]
        if title_rows.empty:
            continue
        title = title_rows.iloc[0]["title"]
        results.append((title, round(float(sim_score), 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n], matched_title  # also return matched title for UI feedback
