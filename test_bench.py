"""
Milestone 3 benchmark script: evaluates our 3 from-scratch models with custom
metrics, 5-fold expanding-window time-series CV, saves milestone3_results.csv,
and creates 3 charts inside plots/.
"""

import csv
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_nba_data, chronological_split, standardize
from models import KNN, GaussianNB, LogisticRegression
from metrics import accuracy, evaluate_model, roc_curve


def expanding_window_cv(make_model, X, y, n_folds=5):
    """
    Time-series cross-validation: always train on earlier games and test on the
    next chunk of future games so no future information leaks into training.
    """
    block = len(y) // (n_folds + 1)
    scores = []
    for fold in range(1, n_folds + 1):
        split = fold * block
        end = (fold + 1) * block if fold < n_folds else len(y)
        X_tr, y_tr = X[:split], y[:split]
        X_val, y_val = X[split:end], y[split:end]
        X_tr_s, X_val_s, _, _ = standardize(X_tr, X_val)

        model = make_model()
        model.fit(X_tr_s, y_tr)
        scores.append(accuracy(y_val, model.predict(X_val_s)))
    return float(np.mean(scores)), float(np.std(scores))


def save_plots(rows, roc_data):
    """Create and save the 3 comparison charts in plots/."""
    os.makedirs("plots", exist_ok=True)
    names = [r["model"] for r in rows]

    # 1. Bar chart comparing Accuracy, F1, ROC-AUC, and 5-Fold CV Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.2
    metrics_to_plot = [
        ("Test Accuracy", [r["accuracy"] for r in rows], "#2b6cb0"),
        ("F1-Score", [r["f1"] for r in rows], "#38a169"),
        ("ROC-AUC", [r["roc_auc"] for r in rows], "#dd6b20"),
        ("5-Fold CV Acc", [r["cv_acc_mean"] for r in rows], "#805ad5"),
    ]
    for i, (label, vals, color) in enumerate(metrics_to_plot):
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("NBA Game Prediction: Model Metrics Comparison (23 Team Features)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/model_metrics_comparison.png", dpi=150)
    plt.close()

    # 2. Confusion matrices for the 3 learned models
    learned = [r for r in rows if r["model"] != "Always Pick Home"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, r in zip(axes, learned):
        mat = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(mat, cmap="Blues")
        ax.set_title(f"{r['model']}\n(Acc: {r['accuracy']*100:.1f}%)")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Away", "Pred Home"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Away", "Actual Home"])
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.55 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrices.png", dpi=150)
    plt.close()

    # 3. ROC curves for the 3 learned models
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (Away Wins Called Home)")
    ax.set_ylabel("True Positive Rate (Home Wins Caught)")
    ax.set_title("ROC Curves on Chronological Test Set (187 Games)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/roc_curves.png", dpi=150)
    plt.close()


def main():
    X, y, _, games = load_nba_data()
    X_train, X_test, y_train, y_test, _, _ = chronological_split(X, y, games)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    # Baseline: always predict home win using the training home-win rate as probability
    base_prob = np.full(len(y_test), float(np.mean(y_train)))
    base_metrics = evaluate_model(y_test, np.ones_like(y_test), base_prob)
    block = len(y) // 6
    base_cv = [float(np.mean(y[f * block : (f + 1) * block if f < 5 else len(y)] == 1))
               for f in range(1, 6)]
    rows = [{"model": "Always Pick Home", **base_metrics,
             "cv_acc_mean": float(np.mean(base_cv)), "cv_acc_std": float(np.std(base_cv)),
             "fit_ms": 0.0, "pred_ms": 0.0}]

    model_factories = {
        "KNN (k=15)": lambda: KNN(k=15),
        "Gaussian NB": lambda: GaussianNB(),
        "Logistic Reg": lambda: LogisticRegression(lr=0.05, epochs=1000),
    }

    roc_data = {}
    for name, make_model in model_factories.items():
        model = make_model()
        t0 = time.perf_counter(); model.fit(X_train_s, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); y_prob = model.predict_proba(X_test_s)
        y_pred = (y_prob >= 0.5).astype(np.int64)
        pred_ms = (time.perf_counter() - t0) * 1000

        m = evaluate_model(y_test, y_pred, y_prob)
        cv_mean, cv_std = expanding_window_cv(make_model, X, y, n_folds=5)
        rows.append({"model": name, **m, "cv_acc_mean": cv_mean, "cv_acc_std": cv_std,
                     "fit_ms": fit_ms, "pred_ms": pred_ms})
        fpr, tpr = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, m["roc_auc"])

    # Print benchmark summary table
    print("=" * 94)
    print("MILESTONE 3 BENCHMARK RESULTS (187 Chronological Test Games + 5-Fold Time-Series CV)")
    print("=" * 94)
    print(f"{'Model':<18} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUC':>7} {'LogLoss':>8} {'5-Fold CV':>11} {'Fit/Pred(ms)':>13}")
    print("-" * 94)
    for r in rows:
        print(f"{r['model']:<18} {r['accuracy']*100:>6.2f}% {r['precision']*100:>6.2f}% "
              f"{r['recall']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['roc_auc']:>7.3f} "
              f"{r['log_loss']:>8.4f} {r['cv_acc_mean']*100:>10.2f}% "
              f"{r['fit_ms']:>6.1f}/{r['pred_ms']:<5.1f}")

    print("\nConfusion Matrices (TN = Away OK, FP = Wrong Home, FN = Missed Home, TP = Home OK):")
    for r in rows:
        print(f"  {r['model']:<18} -> TN: {r['tn']:>3}, FP: {r['fp']:>3}, "
              f"FN: {r['fn']:>3}, TP: {r['tp']:>3}")

    # Save CSV and plots
    with open("results/milestone3_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows, roc_data)
    print("\nSaved milestone3_results.csv and 3 charts inside plots/.")


if __name__ == "__main__":
    main()

