"""
Milestone 2 runner: loads the NBA dataset, trains all 3 from-scratch models,
compares their accuracy against the "always pick home team" baseline, and
prints example predictions from the test set.
"""

import numpy as np
from data_utils import load_nba_data, chronological_split, standardize
from models import KNN, GaussianNB, LogisticRegression


def accuracy(y_true, y_pred):
    """Return the fraction of games predicted correctly."""
    return float(np.mean(y_true == y_pred))


def main():
    # 1. Load, sort chronologically, split 80/20, and scale using only training stats
    X, y, feature_names, games = load_nba_data()
    X_train, X_test, y_train, y_test, games_train, games_test = chronological_split(
        X, y, games, train_ratio=0.8
    )
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    print("=" * 68)
    print("NBA GAME PREDICTION - MILESTONE 2 (FROM SCRATCH)")
    print("=" * 68)
    print(f"Total games : {len(y)}  |  Features used: {len(feature_names)}")
    print(
        f"Train set   : {len(y_train)} games "
        f"({games_train[0]['date']} to {games_train[-1]['date']})"
    )
    print(
        f"Test set    : {len(y_test)} games "
        f"({games_test[0]['date']} to {games_test[-1]['date']})"
    )
    print()

    # 2. Baseline: always predict the home team wins (home_win = 1)
    base_train = accuracy(y_train, np.ones_like(y_train))
    base_test = accuracy(y_test, np.ones_like(y_test))

    # 3. Train and evaluate our 3 from-scratch models
    models = {
        "KNN (k=15)": KNN(k=15),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(lr=0.05, epochs=1000),
    }

    print(f"{'Model':<26} {'Train Acc':>10} {'Test Acc':>10} {'vs Baseline':>13}")
    print("-" * 68)
    print(f"{'Always Pick Home (Base)':<26} {base_train * 100:>9.2f}% {base_test * 100:>9.2f}% {'--':>13}")

    test_probs = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        train_acc = accuracy(y_train, model.predict(X_train_s))
        test_acc = accuracy(y_test, model.predict(X_test_s))
        diff = (test_acc - base_test) * 100
        test_probs[name] = model.predict_proba(X_test_s)
        print(f"{name:<26} {train_acc * 100:>9.2f}% {test_acc * 100:>9.2f}% {diff:>+12.2f}%")

    # 4. Show 5 example predictions from the test set
    print("\n" + "=" * 68)
    print("EXAMPLE TEST SET PREDICTIONS (Predicted Winner & Home-Win Prob)")
    print("=" * 68)

    for idx in range(5):
        g = games_test[idx]
        actual = g["home_abbr"] if g["home_win"] == 1 else g["visitor_abbr"]
        print(f"\n{g['date']} | {g['visitor_abbr']} @ {g['home_abbr']}  -->  Actual Winner: {actual}")

        for name in models:
            prob = test_probs[name][idx]
            pred_team = g["home_abbr"] if prob >= 0.5 else g["visitor_abbr"]
            mark = "OK" if pred_team == actual else "X "
            print(f"  [{mark}] {name:<22}: picks {pred_team} (Home Win Prob: {prob * 100:5.1f}%)")


if __name__ == "__main__":
    main()
