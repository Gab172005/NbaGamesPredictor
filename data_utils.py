"""
Simple data utilities for loading, splitting, and scaling NBA game data.
Uses only Python's built-in csv module and NumPy (no pandas or scikit-learn).
"""

import csv
import numpy as np

# Columns that identify a game or hold the target answer (not model inputs)
NON_FEATURE_COLS = {"game_id", "date", "home_abbr", "visitor_abbr", "home_win"}


def load_nba_data(csv_path="clean/nba_2022_23_prepared.csv"):
    """Load the CSV, sort games in time order, and keep the 23 direct team stats."""
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    # Sort chronologically by game date, using game_id to break same-day ties
    rows.sort(key=lambda row: (row["date"], row["game_id"]))

    # Keep only the 23 intuitive current-season team stats
    # (drop IDs, the target label, and the 36 prior-season *_prev_* player stats)
    feature_names = [
        col for col in columns
        if col not in NON_FEATURE_COLS and "_prev_" not in col
    ]

    # Convert rows into NumPy arrays for our models
    X = np.array(
        [[float(row[col]) for col in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(row["home_win"]) for row in rows], dtype=np.int64)

    # Keep basic game info so we can print readable example predictions later
    games = [
        {
            "game_id": row["game_id"],
            "date": row["date"],
            "home_abbr": row["home_abbr"],
            "visitor_abbr": row["visitor_abbr"],
            "home_win": int(row["home_win"]),
        }
        for row in rows
    ]

    return X, y, feature_names, games


def chronological_split(X, y, games, train_ratio=0.8):
    """Split data in time order: first 80% for training, last 20% for testing."""
    split_idx = int(len(y) * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    games_train, games_test = games[:split_idx], games[split_idx:]

    return X_train, X_test, y_train, y_test, games_train, games_test


def standardize(X_train, X_test):
    """
    Scale each feature to mean 0 and standard deviation 1: (x - mean) / std.
    Mean and std are learned ONLY from X_train so test data never leaks in.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid dividing by zero if a feature has the exact same value in every game
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std
