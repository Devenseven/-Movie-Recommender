"""
data_loader.py
--------------
Handles loading, merging, and transforming the MovieLens dataset.

MovieLens 100k Small Dataset:
  - ratings.csv : userId, movieId, rating (0.5–5.0), timestamp
  - movies.csv  : movieId, title, genres
"""

import pandas as pd
from typing import Tuple


# ─────────────────────────────────────────────
# 1. Raw loaders
# ─────────────────────────────────────────────

def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    # Keep only the columns we need
    df = df[["userId", "movieId", "rating"]].dropna()
    return df


def load_movies(path: str) -> pd.DataFrame:
    """Load movies CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    df = df[["movieId", "title", "genres"]].dropna()
    return df


# ─────────────────────────────────────────────
# 2. Merged view
# ─────────────────────────────────────────────

def merge_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ratings and movies on movieId.

    Returns a DataFrame with columns:
      userId | movieId | rating | title | genres
    """
    merged = pd.merge(ratings, movies, on="movieId", how="inner")
    return merged


# ─────────────────────────────────────────────
# 3. User-Item Matrix (dense pivot)
# ─────────────────────────────────────────────

def build_user_item_matrix(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dense user-item matrix from the merged DataFrame.

    Rows    = users  (userId)
    Columns = movies (title)
    Values  = rating (NaN where user has not rated)
    """
    matrix = merged.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    )
    return matrix


# ─────────────────────────────────────────────
# 4. Train / Test split (no surprise needed)
# ─────────────────────────────────────────────

def train_test_split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split the ratings DataFrame into train and test sets.

    Args:
        ratings:      DataFrame with columns [userId, movieId, rating]
        test_size:    Fraction held out for testing (default 20 %)
        random_state: Seed for reproducibility

    Returns:
        train_df, test_df  – two DataFrames with the same columns.
    """
    test_df = ratings.sample(frac=test_size, random_state=random_state)
    train_df = ratings.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
