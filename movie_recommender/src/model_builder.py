"""
model_builder.py
----------------
Builds and trains collaborative filtering models using pure NumPy / scikit-learn
(no scikit-surprise required — compatible with Python 3.13+).

SVD (Singular Value Decomposition) factorizes the user-item rating matrix R
into latent factor matrices:

    R ≈ U · Σ · Vᵀ

Where:
  - U  : user latent factor matrix  (users × factors)
  - Σ  : diagonal matrix of singular values
  - Vᵀ : item latent factor matrix  (factors × items)

Predicted rating for user u on item i:
    r̂(u, i) = μ + b_u + b_i + q_i · pᵤᵀ

Where μ = global mean, b_u/b_i = user/item biases, p_u/q_i = latent vectors.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


# ─────────────────────────────────────────────
# Custom SVD Model (SGD with biases)
# ─────────────────────────────────────────────

class SVDModel:
    """
    Matrix-factorisation SVD trained with Stochastic Gradient Descent.

    Replicates the core algorithm used by scikit-surprise's SVD class:
        r̂(u, i) = μ + b_u + b_i + p_u · q_iᵀ

    Parameters
    ----------
    n_factors : int
        Number of latent dimensions.
    n_epochs : int
        SGD passes over the training data.
    lr : float
        Learning rate.
    reg : float
        L2 regularisation weight.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        # set after fit()
        self.global_mean: float = 0.0
        self.bu: np.ndarray | None = None   # user biases
        self.bi: np.ndarray | None = None   # item biases
        self.pu: np.ndarray | None = None   # user latent factors  (n_users × n_factors)
        self.qi: np.ndarray | None = None   # item latent factors  (n_items × n_factors)

        # Internal mappings built during fit
        self._user_to_idx: dict = {}
        self._item_to_idx: dict = {}
        self._idx_to_item: dict = {}

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        """
        Train on a DataFrame with columns [userId, movieId, rating].

        Returns self for chaining.
        """
        rng = np.random.default_rng(self.random_state)

        users = train_df["userId"].unique()
        items = train_df["movieId"].unique()

        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: it for it, i in self._item_to_idx.items()}

        n_users = len(users)
        n_items = len(items)

        self.global_mean = float(train_df["rating"].mean())

        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.pu = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.qi = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Pre-map indices once for speed
        u_idx = train_df["userId"].map(self._user_to_idx).to_numpy()
        i_idx = train_df["movieId"].map(self._item_to_idx).to_numpy()
        ratings = train_df["rating"].to_numpy(dtype=float)
        order = np.arange(len(ratings))

        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            for k in order:
                u, i, r = u_idx[k], i_idx[k], ratings[k]
                err = r - (self.global_mean + self.bu[u] + self.bi[i]
                           + self.pu[u] @ self.qi[i])

                # Bias updates
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # Latent factor updates
                pu_old = self.pu[u].copy()
                self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                self.qi[i] += self.lr * (err * pu_old       - self.reg * self.qi[i])

        return self

    # ------------------------------------------------------------------
    def predict(self, user_id, movie_id) -> float:
        """
        Predict the rating for (user_id, movie_id).

        Falls back to the global mean for unseen users / items.
        """
        mu = self.global_mean
        bu = self.bu[self._user_to_idx[user_id]] if user_id in self._user_to_idx else 0.0
        bi = self.bi[self._item_to_idx[movie_id]] if movie_id in self._item_to_idx else 0.0

        if user_id in self._user_to_idx and movie_id in self._item_to_idx:
            dot = self.pu[self._user_to_idx[user_id]] @ self.qi[self._item_to_idx[movie_id]]
        else:
            dot = 0.0

        return float(np.clip(mu + bu + bi + dot, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return an array of predicted ratings for each row of test_df."""
        return np.array(
            [self.predict(r.userId, r.movieId) for r in test_df.itertuples()]
        )

    def to_inner_iid(self, raw_iid):
        """Map raw movie ID → internal index (mirrors surprise API for compat)."""
        if raw_iid not in self._item_to_idx:
            raise ValueError(f"Item {raw_iid} not in trainset.")
        return self._item_to_idx[raw_iid]

    def to_raw_iid(self, inner_id):
        """Map internal index → raw movie ID."""
        return self._idx_to_item[inner_id]

    @property
    def trainset(self):
        """Expose self so recommend.py can call algo.trainset.to_inner_iid."""
        return self


# ─────────────────────────────────────────────
# Public builder functions (mirrors old API)
# ─────────────────────────────────────────────

def get_traintest_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the ratings DataFrame into train and test sets.

    Returns:
        train_df, test_df
    """
    from src.data_loader import train_test_split_ratings
    return train_test_split_ratings(ratings, test_size=test_size, random_state=random_state)


def train_svd_model(
    train_df: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
) -> SVDModel:
    """
    Train an SVDModel on the given training DataFrame.

    Args:
        train_df:  DataFrame with [userId, movieId, rating]
        n_factors: Number of latent factors
        n_epochs:  SGD passes

    Returns:
        Fitted SVDModel
    """
    model = SVDModel(n_factors=n_factors, n_epochs=n_epochs)
    model.fit(train_df)
    return model


def train_knn_model(train_df: pd.DataFrame) -> SVDModel:
    """
    Train an item-based KNN model.

    We re-use SVDModel here because the item latent vectors (qi) already
    encode item similarity perfectly for cosine-similarity KNN lookup.
    The recommend.py `get_similar_movies` function uses those vectors directly.

    Args:
        train_df: DataFrame with [userId, movieId, rating]

    Returns:
        Fitted SVDModel (item vectors used for KNN lookup)
    """
    return train_svd_model(train_df, n_factors=50, n_epochs=20)
