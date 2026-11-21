# NBA Game Outcome Predictor (Built From Scratch)

A beginner-friendly machine learning project that predicts whether the **home team wins** an NBA regular-season game (`home_win`: `1` = Home Win, `0` = Away Win) using **three algorithms built 100% from scratch** with only Python's standard library (`csv`, `math`) and `numpy` — no `scikit-learn` or `pandas`.

---

## Project Overview

- **Goal:** Predict NBA game outcomes before tipoff using easy-to-explain team performance, momentum, and schedule-fatigue stats, and compare three classic ML algorithms fairly without any data leakage.
- **Dataset:** `clean/nba_2022_23_prepared.csv` — **931 regular-season games** from the 2022–23 NBA season (`2022-10-18` to `2023-02-28`, 0 missing values).
- **Features Used (23 Direct Team Stats):** We dropped the 36 prior-season player-stat columns (`*_prev_*`) and kept only the 23 clear, current-season pre-game team stats:
  - **Season Record & Games Played (5):** `home_games_played`, `visitor_games_played`, `home_win_pct`, `visitor_win_pct`, `diff_win_pct`
  - **Scoring & Point Differential (7):** `home_avg_point_diff`, `visitor_avg_point_diff`, `diff_avg_point_diff`, `home_avg_pts_scored`, `visitor_avg_pts_scored`, `home_avg_pts_allowed`, `visitor_avg_pts_allowed`
  - **Recent 5-Game Form (6):** `home_last5_win_pct`, `visitor_last5_win_pct`, `diff_last5_win_pct`, `home_last5_point_diff`, `visitor_last5_point_diff`, `diff_last5_point_diff`
  - **Rest & Fatigue (5):** `home_rest_days`, `visitor_rest_days`, `diff_rest_days`, `home_is_b2b`, `visitor_is_b2b`
- **Chronological 80/20 Split (No Random Shuffling):**
  - **Training Set (first 80%):** 744 games (`2022-10-18` to `2023-01-28`) — 59.95% home wins
  - **Test Set (last 20%):** 187 games (`2023-01-28` to `2023-02-28`) — 58.29% home wins
  - Sorting by `(date, game_id)` ensures we only ever use past games to predict future games. Features are standardized with `(x - mean) / std` computed **strictly on the training set**.
- **Baseline Benchmark:** **Always Pick Home Team** (`58.29%` test accuracy), reflecting natural NBA home-court advantage.

---

## Current Progress: Milestone 2 — Data Pipeline & Three From-Scratch Models

**Files:** `data_utils.py`, `models.py`, `run_milestone2.py`, `milestone_2.txt`

In Milestone 2, we built the complete loading, chronological splitting, and scaling pipeline along with all three from-scratch models in one clean file (`models.py`), each supporting `.fit(X, y)`, `.predict_proba(X)`, and `.predict(X)`:

1. **K-Nearest Neighbors (`KNN`, default `k=15`):** Stores the scaled training games. For a new game, it calculates the straight-line (Euclidean) distance to every past game, finds the 15 most similar matchups, and estimates the home-win probability by taking a vote among them.
2. **Gaussian Naive Bayes (`GaussianNB`, default `var_smoothing=1e-6`):** Computes the mean and variance of each stat separately for past home wins vs. home losses. For a new game, it compares how well the stats match each bell curve (in log-space to avoid tiny-number underflow) and picks the higher-probability winner.
3. **Logistic Regression (`LogisticRegression`, default `lr=0.05, epochs=1000`):** Learns one weight per stat plus a bias term using simple batch gradient descent, then passes the weighted sum through a sigmoid curve to output a 0%–100% home-win probability.

### Initial Baseline Results (`python run_milestone2.py`)

| Model | Train Accuracy | Test Accuracy | vs. Baseline (Test) |
| :--- | :---: | :---: | :---: |
| Always Pick Home (Baseline) | 59.95% | 58.29% | -- |
| `KNN (k=15)` | 65.46% | 63.10% | +4.81% |
| `Gaussian Naive Bayes` | 61.29% | 62.57% | +4.28% |
| `Logistic Regression` | 62.77% | 63.10% | +4.81% |

**Key Takeaways:**
- All three untuned models beat the "always pick home" baseline by more than 4 percentage points on unseen late-season games.
- `KNN (k=15)` and `Logistic Regression` tied for the highest initial test accuracy (`63.10%`), with `Gaussian Naive Bayes` close behind (`62.57%`).
- Train and test accuracies stay close across all three models, suggesting they capture genuine basketball signal with minimal overfitting.
- Running `run_milestone2.py` also prints 5 example test-set matchups with each model's predicted winner and home-win probability.

---

## Next Steps (Upcoming Milestones)

- **Milestone 3:** Implement deeper from-scratch evaluation metrics (confusion matrices, precision, recall, F1-score, ROC-AUC, and log-loss), time-series cross-validation, and diagnostic plots.
- **Milestone 4:** Perform leak-free hyperparameter tuning strictly on the training set and analyze final model performance and feature importance.

---

## Project Structure

```text
NbaGamesPredictor/
├── clean/
│   ├── nba_2022_23_prepared.csv    # Prepared 931-game dataset used by all scripts
│   ├── nba_games_clean.csv         # Cleaned raw game scores
│   └── all_seasons_clean.csv       # Historical player-season stats
├── data_utils.py                   # CSV loader, 23-feature selector, 80/20 split, scaler
├── models.py                       # From-scratch KNN, GaussianNB, LogisticRegression
├── run_milestone2.py               # Milestone 2: train/test accuracy & example predictions
├── milestone_2.txt                 # Plain-English Milestone 2 summary
└── README.md                       # Project documentation
```

---

## How to Run

Requires Python 3.10+ with `numpy` installed:

```bash
# Milestone 2: Train all 3 models, compare against baseline, and view example test predictions
python run_milestone2.py
```
