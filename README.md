# 🎬 Movie Recommendation System

A complete, production-quality Movie Recommendation System built with **Collaborative Filtering (SVD)** using the [MovieLens 100k Small Dataset](https://grouplens.org/datasets/movielens/latest/) and served via a **Streamlit** web UI.

> **Python 3.13 compatible** — uses a pure NumPy/scikit-learn SVD implementation (no scikit-surprise needed).

---

## 📁 Project Structure

```
movie_recommender/
├── data/
│   ├── ratings.csv       # userId, movieId, rating, timestamp
│   └── movies.csv        # movieId, title, genres
├── src/
│   ├── data_loader.py    # Data loading & train/test split
│   ├── model_builder.py  # Custom SVD model (NumPy SGD)
│   ├── recommend.py      # Recommendation & similarity logic
│   └── evaluate.py       # RMSE evaluation pipeline
├── app.py                # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## � How to Run

### Step 1 — Install dependencies

Open a terminal (PowerShell) and run:

```powershell
pip install pandas scikit-learn numpy streamlit
```

Or using the requirements file:

```powershell
pip install -r requirements.txt
```

### Step 2 — Launch the web app

Run this command from inside the `movie_recommender/` folder:

```powershell
py -3.13 -m streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**.

> ⏳ **First load takes ~30 seconds** while the SVD model trains. After that it's instant (cached for the session).

---

## 🖥️ Run the Evaluation Script (optional)

To just print the RMSE score in the terminal without the UI:

```powershell
py -3.13 src/evaluate.py
```

---

## 🎮 How to Use the App

The app has **3 tabs**:

| Tab | What it does |
|-----|--------------|
| 👤 **User Recommendations** | Pick a User ID from the dropdown → click **Recommend for User** → see top-N movie picks with predicted ratings |
| 🎭 **Similar Movies** | Type any movie title (typos OK) → click **Find Similar Movies** → see the most similar films |
| 📊 **Model Evaluation** | Shows the RMSE score and explains how SVD works |

Use the **sidebar slider** to control how many recommendations (3–20) are shown.

---

## 📊 Dataset

The **MovieLens 100k Small** dataset contains:
- **~100,000 ratings** from **610 users** on **9,742 movies**
- Ratings are on a **0.5 – 5.0** scale (in 0.5 increments)
- Source: [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)

---

## 📐 RMSE — What It Means

**Root Mean Squared Error** measures prediction accuracy:

```
RMSE = sqrt( (1/N) * Σ (r_ui - r̂_ui)² )
```

| Value | Interpretation |
|-------|---------------|
| < 0.85 | Excellent |
| 0.85–0.93 | Good (typical SVD range on ML-100k) |
| > 1.0 | Poor |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | SVD model & numerical operations |
| `scikit-learn` | Cosine similarity for movie lookup |
| `streamlit` | Web UI |

---

## 🛡️ Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Unknown user ID | Clear error message with valid ID range |
| Typo in movie title | Fuzzy-matched via Ratcliff/Obershelp algorithm |
| Movie with no ratings | Informative error message |
| User has rated all movies | Informative error message |
