"""
Milestone 4 hyperparameter tuning using 5-fold expanding-window time-series CV.
IMPORTANT: Only the first 744 training games are used here. The final 187 test
games stay completely untouched ("in a vault") so there is zero data leakage.
"""

import csv
import json
import numpy as np

from data_utils import load_nba_data, chronological_split, standardize
from models import KNN, GaussianNB, LogisticRegression
from metrics import accuracy, roc_auc, log_loss


def train_cv_score(make_model, X_train, y_train, n_folds=5):
    """Evaluate a model setting across 5 expanding chronological training folds."""
    block = len(y_train) // (n_folds + 1)  # 744 // 6 = 124 games per block
    accs, aucs, losses = [], [], []

    for fold in range(1, n_folds + 1):
        split = fold * block
        end = (fold + 1) * block if fold < n_folds else len(y_train)

        X_tr, y_tr = X_train[:split], y_train[:split]
        X_val, y_val = X_train[split:end], y_train[split:end]
        X_tr_s, X_val_s, _, _ = standardize(X_tr, X_val)

        model = make_model()
        model.fit(X_tr_s, y_tr)
        probs = model.predict_proba(X_val_s)
        preds = (probs >= 0.5).astype(np.int64)

        accs.append(accuracy(y_val, preds))
        aucs.append(roc_auc(y_val, probs))
        losses.append(log_loss(y_val, probs))

    return float(np.mean(accs)), float(np.mean(aucs)), float(np.mean(losses))


def main():
    X, y, _, games = load_nba_data()
    X_train, _, y_train, _, games_train, _ = chronological_split(X, y, games)

    print("=" * 74)
    print("MILESTONE 4: HYPERPARAMETER TUNING (STRICTLY ON 744 TRAINING GAMES)")
    print(f"Training range: {games_train[0]['date']} to {games_train[-1]['date']}")
    print("=" * 74)

    # Simple, easy-to-explain grids for all 3 models
    search_space = {
        "KNN": [
            ({"k": k}, lambda k=k: KNN(k=k))
            for k in [5, 9, 15, 21, 27, 35, 45, 55]
        ],
        "Gaussian NB": [
            ({"var_smoothing": vs}, lambda vs=vs: GaussianNB(var_smoothing=vs))
            for vs in [1e-9, 1e-6, 1e-3, 0.05, 0.2, 0.5, 1.0, 2.0]
        ],
        "Logistic Reg": [
            (
                {"lr": lr, "epochs": ep, "l2": l2},
                lambda lr=lr, ep=ep, l2=l2: LogisticRegression(lr=lr, epochs=ep, l2=l2),
            )
            for lr in [0.01, 0.05, 0.1]
            for ep in [300, 1000]
            for l2 in [0.0, 0.01, 0.1, 0.5, 1.0]
        ],
    }

    audit_rows = []
    best_params = {}

    for model_name, configs in search_space.items():
        best_key = None
        best_cfg = None
        best_scores = None

        for params, make_model in configs:
            cv_acc, cv_auc, cv_loss = train_cv_score(make_model, X_train, y_train)
            audit_rows.append({
                "model": model_name,
                "params": json.dumps(params),
                "train_cv_acc": round(cv_acc, 6),
                "train_cv_auc": round(cv_auc, 6),
                "train_cv_logloss": round(cv_loss, 6),
            })

            # Pick the setting with the highest CV accuracy (tie-break: ROC-AUC, then Log-Loss)
            score_key = (round(cv_acc, 6), round(cv_auc, 6), -round(cv_loss, 6))
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_cfg = params
                best_scores = (cv_acc, cv_auc, cv_loss)

        best_params[model_name] = best_cfg
        print(
            f"{model_name:<14} | Best: {str(best_cfg):<36} | "
            f"CV Acc: {best_scores[0]*100:5.2f}% | AUC: {best_scores[1]:.3f} | "
            f"LogLoss: {best_scores[2]:.4f}"
        )

    # Save full grid-search log and winning settings
    with open("tuning_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    with open("best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nTested {len(audit_rows)} configs total.")
    print("Saved tuning_results.csv and best_hyperparams.json.")


if __name__ == "__main__":
    main()

