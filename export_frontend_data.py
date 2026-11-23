"""
Export trained from-scratch NBA models, team snapshots, benchmark metrics,
and holdout test games into frontend/model_data.json for the static
GitHub Pages web app.

Uses only Python's standard library, NumPy, and the project's existing
data_utils.py and models.py (no scikit-learn or pandas).
"""

import csv
import json
import os
import numpy as np

from data_utils import load_nba_data, chronological_split, standardize
from models import KNN, GaussianNB, LogisticRegression


TEAM_INFO = {
    "ATL": ("Atlanta Hawks", "East", "#c8102e"),
    "BKN": ("Brooklyn Nets", "East", "#1f2937"),
    "BOS": ("Boston Celtics", "East", "#007a33"),
    "CHA": ("Charlotte Hornets", "East", "#00788c"),
    "CHI": ("Chicago Bulls", "East", "#ce1141"),
    "CLE": ("Cleveland Cavaliers", "East", "#860038"),
    "DAL": ("Dallas Mavericks", "West", "#00538c"),
    "DEN": ("Denver Nuggets", "West", "#0e2240"),
    "DET": ("Detroit Pistons", "East", "#c8102e"),
    "GSW": ("Golden State Warriors", "West", "#1d428a"),
    "HOU": ("Houston Rockets", "West", "#ce1141"),
    "IND": ("Indiana Pacers", "East", "#002d62"),
    "LAC": ("LA Clippers", "West", "#c8102e"),
    "LAL": ("Los Angeles Lakers", "West", "#552583"),
    "MEM": ("Memphis Grizzlies", "West", "#5d76a9"),
    "MIA": ("Miami Heat", "East", "#98002e"),
    "MIL": ("Milwaukee Bucks", "East", "#00471b"),
    "MIN": ("Minnesota Timberwolves", "West", "#0c2340"),
    "NOP": ("New Orleans Pelicans", "West", "#0c2340"),
    "NYK": ("New York Knicks", "East", "#006bb6"),
    "OKC": ("Oklahoma City Thunder", "West", "#007ac1"),
    "ORL": ("Orlando Magic", "East", "#0077c0"),
    "PHI": ("Philadelphia 76ers", "East", "#006bb6"),
    "PHX": ("Phoenix Suns", "West", "#1d1160"),
    "POR": ("Portland Trail Blazers", "West", "#e03a3e"),
    "SAC": ("Sacramento Kings", "West", "#5a2d81"),
    "SAS": ("San Antonio Spurs", "West", "#374151"),
    "TOR": ("Toronto Raptors", "East", "#ce1141"),
    "UTA": ("Utah Jazz", "West", "#002b5c"),
    "WAS": ("Washington Wizards", "East", "#002b5c"),
}


def load_game_scores(csv_path="clean/nba_games_clean.csv"):
    """Load actual home/visitor final scores keyed by game_id."""
    scores = {}
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores[row["game_id"]] = (int(row["home_pts"]), int(row["visitor_pts"]))
    return scores


def load_benchmarks(csv_path="results/milestone4_results.csv"):
    """Load Milestone 4 benchmark table into clean JSON records."""
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "label": r["label"],
                "model": r["model"],
                "variant": r["variant"],
                "params": r["params"],
                "accuracy": round(float(r["accuracy"]), 6),
                "precision": round(float(r["precision"]), 6),
                "recall": round(float(r["recall"]), 6),
                "f1": round(float(r["f1"]), 6),
                "roc_auc": round(float(r["roc_auc"]), 6),
                "log_loss": round(float(r["log_loss"]), 6),
                "tn": int(r["tn"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
                "tp": int(r["tp"]),
                "cv_acc_mean": round(float(r["cv_acc_mean"]), 6),
                "cv_acc_std": round(float(r["cv_acc_std"]), 6),
            })
    return rows


def extract_team_snapshots(X, feature_names, games):
    """Grab each team's most recent 2022-23 pre-tipoff stats from the dataset."""
    idx = {name: i for i, name in enumerate(feature_names)}
    latest = {}

    for row_vals, g in zip(X, games):
        latest[g["home_abbr"]] = {
            "games_played": int(round(row_vals[idx["home_games_played"]])),
            "win_pct": round(float(row_vals[idx["home_win_pct"]]), 3),
            "last5_win_pct": round(float(row_vals[idx["home_last5_win_pct"]]), 2),
            "avg_pts_scored": round(float(row_vals[idx["home_avg_pts_scored"]]), 1),
            "avg_pts_allowed": round(float(row_vals[idx["home_avg_pts_allowed"]]), 1),
            "avg_point_diff": round(float(row_vals[idx["home_avg_point_diff"]]), 1),
            "last5_point_diff": round(float(row_vals[idx["home_last5_point_diff"]]), 1),
            "as_of": g["date"],
        }
        latest[g["visitor_abbr"]] = {
            "games_played": int(round(row_vals[idx["visitor_games_played"]])),
            "win_pct": round(float(row_vals[idx["visitor_win_pct"]]), 3),
            "last5_win_pct": round(float(row_vals[idx["visitor_last5_win_pct"]]), 2),
            "avg_pts_scored": round(float(row_vals[idx["visitor_avg_pts_scored"]]), 1),
            "avg_pts_allowed": round(float(row_vals[idx["visitor_avg_pts_allowed"]]), 1),
            "avg_point_diff": round(float(row_vals[idx["visitor_avg_point_diff"]]), 1),
            "last5_point_diff": round(float(row_vals[idx["visitor_last5_point_diff"]]), 1),
            "as_of": g["date"],
        }

    teams = []
    for abbr in sorted(latest.keys()):
        full_name, conference, color = TEAM_INFO.get(abbr, (abbr, "NBA", "#1d4ed8"))
        teams.append({
            "abbr": abbr,
            "name": full_name,
            "conference": conference,
            "color": color,
            "stats": latest[abbr],
        })
    return teams


def round_list(arr, decimals=6):
    return [round(float(v), decimals) for v in arr]


def main():
    X, y, feature_names, games = load_nba_data()
    X_train, X_test, y_train, y_test, games_train, games_test = chronological_split(
        X, y, games, train_ratio=0.8
    )
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)

    with open("results/best_hyperparams.json", mode="r", encoding="utf-8") as f:
        best = json.load(f)

    # Train all 3 tuned models identically to run_milestone4.py
    gnb = GaussianNB(**best["Gaussian NB"]).fit(X_train_s, y_train)
    lr = LogisticRegression(**best["Logistic Reg"]).fit(X_train_s, y_train)
    knn = KNN(**best["KNN"]).fit(X_train_s, y_train)

    # Generate reference probabilities on the 187 holdout test games
    gnb_probs = gnb.predict_proba(X_test_s)
    lr_probs = lr.predict_proba(X_test_s)
    knn_probs = knn.predict_proba(X_test_s)

    scores = load_game_scores()
    test_games = []
    for i, g in enumerate(games_test):
        home_pts, visitor_pts = scores.get(g["game_id"], (0, 0))
        test_games.append({
            "id": g["game_id"],
            "date": g["date"],
            "home": g["home_abbr"],
            "visitor": g["visitor_abbr"],
            "homePts": home_pts,
            "visitorPts": visitor_pts,
            "homeWin": int(g["home_win"]),
            "features": round_list(X_test[i], 6),
            "probs": {
                "gnb": round(float(gnb_probs[i]), 6),
                "lr": round(float(lr_probs[i]), 6),
                "knn": round(float(knn_probs[i]), 6),
            },
        })

    # Compact historical game labels for KNN neighbor inspection
    train_labels = []
    for g in games_train:
        h_pts, v_pts = scores.get(g["game_id"], (0, 0))
        train_labels.append(
            f"{g['date']}|{g['visitor_abbr']}@{g['home_abbr']}|{v_pts}-{h_pts}|{g['home_win']}"
        )

    bundle = {
        "metadata": {
            "totalGames": int(len(y)),
            "trainGames": int(len(y_train)),
            "testGames": int(len(y_test)),
            "trainDateRange": [games_train[0]["date"], games_train[-1]["date"]],
            "testDateRange": [games_test[0]["date"], games_test[-1]["date"]],
            "homeWinRateTrain": round(float(np.mean(y_train)), 6),
            "homeWinRateTest": round(float(np.mean(y_test)), 6),
            "featureNames": feature_names,
        },
        "scaler": {
            "mean": round_list(mean, 8),
            "std": round_list(std, 8),
        },
        "models": {
            "gaussian_nb": {
                "params": best["Gaussian NB"],
                "means": {
                    "0": round_list(gnb.means[0], 8),
                    "1": round_list(gnb.means[1], 8),
                },
                "vars": {
                    "0": round_list(gnb.vars[0], 8),
                    "1": round_list(gnb.vars[1], 8),
                },
                "priors": {
                    "0": round(float(gnb.priors[0]), 8),
                    "1": round(float(gnb.priors[1]), 8),
                },
            },
            "logistic_regression": {
                "params": best["Logistic Reg"],
                "weights": round_list(lr.weights, 8),
                "bias": round(float(lr.bias), 8),
            },
            "knn": {
                "params": best["KNN"],
                "X_train": [round_list(row, 4) for row in X_train_s],
                "y_train": [int(v) for v in y_train],
                "train_games": train_labels,
            },
        },
        "teams": extract_team_snapshots(X, feature_names, games),
        "benchmarks": load_benchmarks(),
        "testGames": test_games,
    }

    os.makedirs("frontend", exist_ok=True)
    out_path = os.path.join("frontend", "model_data.json")
    with open(out_path, mode="w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported {out_path} ({size_kb:.1f} KB)")
    print(f"  Teams       : {len(bundle['teams'])}")
    print(f"  Train games : {len(bundle['models']['knn']['y_train'])}")
    print(f"  Test games  : {len(bundle['testGames'])}")
    print(f"  Benchmarks  : {len(bundle['benchmarks'])} rows")


if __name__ == "__main__":
    main()
