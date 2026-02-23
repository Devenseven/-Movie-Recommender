"""
evaluate.py
-----------
Evaluates the collaborative filtering model using RMSE on a held-out test set.

RMSE (Root Mean Squared Error):
    RMSE = sqrt( (1/N) * Σ (r_ui - r̂_ui)² )

Where:
  r_ui   = true rating given by user u to item i
  r̂_ui  = predicted rating from the model
  N      = number of test samples

A lower RMSE means more accurate predictions.
Typical SVD RMSE on MovieLens 100k ≈ 0.87–0.93.
"""

import os
import sys
import numpy as np

# Allow running this script directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_ratings, train_test_split_ratings
from src.model_builder import train_svd_model


def evaluate_model(algo, test_df) -> float:
    """
    Compute RMSE of a trained SVDModel on the test DataFrame.

    Args:
        algo:     Trained SVDModel
        test_df:  DataFrame with columns [userId, movieId, rating]

    Returns:
        RMSE score as a float.
    """
    preds = algo.predict_batch(test_df)
    actuals = test_df["rating"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    print(f"RMSE: {rmse:.4f}")
    return rmse


def run_evaluation(
    ratings_path: str = "data/ratings.csv",
    n_factors: int = 100,
    n_epochs: int = 20,
) -> float:
    """
    Full pipeline: load data → split → train → evaluate.

    Args:
        ratings_path: Path to the MovieLens ratings.csv file
        n_factors:    SVD latent factor count
        n_epochs:     Training epochs

    Returns:
        RMSE score.
    """
    print("=" * 50)
    print("  Movie Recommender — RMSE Evaluation")
    print("=" * 50)

    print(f"\n[1/4] Loading ratings from '{ratings_path}' ...")
    ratings = load_ratings(ratings_path)
    print(f"      {len(ratings):,} ratings loaded for "
          f"{ratings['userId'].nunique()} users and "
          f"{ratings['movieId'].nunique()} movies.")

    print("\n[2/4] Splitting into train (80%) / test (20%) ...")
    train_df, test_df = train_test_split_ratings(ratings, test_size=0.2)
    print(f"      Train size: {len(train_df):,} | Test size: {len(test_df):,}")

    print(f"\n[3/4] Training SVD (n_factors={n_factors}, n_epochs={n_epochs}) ...")
    algo = train_svd_model(train_df, n_factors=n_factors, n_epochs=n_epochs)

    print("\n[4/4] Evaluating ...")
    rmse = evaluate_model(algo, test_df)
    print(f"\n✅  SVD RMSE on MovieLens 100k test set: {rmse:.4f}")
    print("    (Lower is better; typical range: 0.87–0.93)\n")

    return rmse


if __name__ == "__main__":
    run_evaluation()
