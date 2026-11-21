"""
Milestone 4 final evaluation script: loads the best hyperparameters chosen by
tune_models.py, compares Untuned vs. Tuned models on the held-out 187 test
games + 5-fold time-series CV, saves milestone4_results.csv, and draws the
final charts in plots/.
"""

import csv
import json
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_nba_data, chronological_split, standardize
from models import KNN, GaussianNB, LogisticRegression
from metrics import accuracy, evaluate_model


def expanding_window_cv(make_model, X, y, n_folds=5):
    """5-fold expanding-window time-series CV across the full 931-game season."""
    block = len(y) // (n_folds + 1)
    scores = []
    for fold in range(1, n_folds + 1):
        split = fold * block
        end = (fold + 1) * block if fold < n_folds else len(y)
        X_tr_s, X_val_s, _, _ = standardize(X[:split], X[split:end])
        model = make_model()
        model.fit(X_tr_s, y[:split])
        scores.append(accuracy(y[split:end], model.predict(X_val_s)))
    return float(np.mean(scores)), float(np.std(scores))


def save_final_plots(rows, feature_names, lr_weights):
    """Create the 3 Milestone 4 presentation charts inside plots/."""
    os.makedirs("plots", exist_ok=True)

    # 1. Untuned vs. Tuned comparison (Accuracy, ROC-AUC, Log-Loss)
    model_labels = ["KNN", "Gaussian NB", "Logistic Reg"]
    untuned = [r for r in rows if r["variant"] == "Untuned"]
    tuned = [r for r in rows if r["variant"] == "Tuned"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    chart_specs = [
        ("Test Accuracy (Higher Better)", "accuracy", (0.50, 0.72)),
        ("ROC-AUC (Higher Better)", "roc_auc", (0.50, 0.76)),
        ("Log-Loss (Lower Better)", "log_loss", (0.55, 0.76)),
    ]
    x = np.arange(len(model_labels))
    w = 0.34
    for ax, (title, key, ylim) in zip(axes, chart_specs):
        u_vals = [r[key] for r in untuned]
        t_vals = [r[key] for r in tuned]
        b1 = ax.bar(x - w / 2, u_vals, width=w, label="Untuned (M3)", color="#a0aec0")
        b2 = ax.bar(x + w / 2, t_vals, width=w, label="Tuned (M4)", color="#2b6cb0")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(model_labels)
        ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")
    plt.suptitle("Milestone 4: Untuned vs. Tuned Performance (187 Holdout Games)", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/tuned_vs_untuned.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Top 10 Basketball Stats driving Tuned Logistic Regression
    top_idx = np.argsort(np.abs(lr_weights))[-10:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = lr_weights[top_idx]
    colors = ["#2b6cb0" if v >= 0 else "#dd6b20" for v in top_vals]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(top_names, top_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Learned Weight (Blue = Helps Home Win, Orange = Helps Away Win)")
    ax.set_title("Top 10 Most Important NBA Stats (Tuned Logistic Regression)")
    pad = max(abs(top_vals)) * 0.03
    for b, v in zip(bars, top_vals):
        ha = "left" if v >= 0 else "right"
        ax.text(v + (pad if v >= 0 else -pad), b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", ha=ha, fontsize=8.5)
    ax.set_xlim(min(top_vals) * 1.45, max(top_vals) * 1.28)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()

    # 3. Final comparison slide: Baseline + 3 Tuned Models
    final_rows = [rows[0]] + tuned
    names = [r["label"] for r in final_rows]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(names))
    w = 0.2
    specs = [
        ("Test Accuracy", "accuracy", "#2b6cb0"),
        ("F1-Score", "f1", "#38a169"),
        ("ROC-AUC", "roc_auc", "#dd6b20"),
        ("5-Fold CV Acc", "cv_acc_mean", "#805ad5"),
    ]
    for i, (label, key, color) in enumerate(specs):
        vals = [r[key] for r in final_rows]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 0.90); ax.set_ylabel("Score")
    ax.set_title("Final NBA Model Comparison: Home Baseline vs. Tuned From-Scratch Models")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/final_model_comparison.png", dpi=150)
    plt.close()


def main():
    X, y, feature_names, games = load_nba_data()
    X_train, X_test, y_train, y_test, _, _ = chronological_split(X, y, games)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    with open("results/best_hyperparams.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    # Home-team baseline
    base_prob = np.full(len(y_test), float(np.mean(y_train)))
    base_m = evaluate_model(y_test, np.ones_like(y_test), base_prob)
    block = len(y) // 6
    base_cv = [float(np.mean(y[f * block : (f + 1) * block if f < 5 else len(y)] == 1))
               for f in range(1, 6)]
    rows = [{"label": "Always Pick Home", "model": "Baseline", "variant": "Baseline",
             "params": "home_win=1", **base_m,
             "cv_acc_mean": float(np.mean(base_cv)), "cv_acc_std": float(np.std(base_cv)),
             "fit_ms": 0.0, "pred_ms": 0.0}]

    # Untuned (Milestone 3 defaults) vs. Tuned (Milestone 4 train-CV winners)
    specs = [
        ("KNN", "Untuned", {"k": 15}, lambda: KNN(k=15)),
        ("KNN", "Tuned", best["KNN"], lambda: KNN(**best["KNN"])),
        ("Gaussian NB", "Untuned", {"var_smoothing": 1e-6}, lambda: GaussianNB()),
        ("Gaussian NB", "Tuned", best["Gaussian NB"], lambda: GaussianNB(**best["Gaussian NB"])),
        ("Logistic Reg", "Untuned", {"lr": 0.05, "epochs": 1000, "l2": 0.0},
         lambda: LogisticRegression(lr=0.05, epochs=1000, l2=0.0)),
        ("Logistic Reg", "Tuned", best["Logistic Reg"],
         lambda: LogisticRegression(**best["Logistic Reg"])),
    ]

    tuned_lr_weights = None
    for name, variant, params, make_model in specs:
        model = make_model()
        t0 = time.perf_counter(); model.fit(X_train_s, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); y_prob = model.predict_proba(X_test_s)
        y_pred = (y_prob >= 0.5).astype(np.int64)
        pred_ms = (time.perf_counter() - t0) * 1000

        m = evaluate_model(y_test, y_pred, y_prob)
        cv_mean, cv_std = expanding_window_cv(make_model, X, y, n_folds=5)
        rows.append({
            "label": f"{name} ({variant})", "model": name, "variant": variant,
            "params": json.dumps(params), **m,
            "cv_acc_mean": cv_mean, "cv_acc_std": cv_std,
            "fit_ms": fit_ms, "pred_ms": pred_ms,
        })
        if name == "Logistic Reg" and variant == "Tuned":
            tuned_lr_weights = model.weights

    print("=" * 98)
    print("MILESTONE 4 FINAL RESULTS: UNTUNED VS. TUNED (187 HOLDOUT GAMES + 5-FOLD CV)")
    print("=" * 98)
    print(f"{'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUC':>7} {'LogLoss':>8} {'5-Fold CV':>11} {'TN/FP/FN/TP':>14}")
    print("-" * 98)
    for r in rows:
        cm = f"{r['tn']}/{r['fp']}/{r['fn']}/{r['tp']}"
        print(f"{r['label']:<24} {r['accuracy']*100:>6.2f}% {r['precision']*100:>6.2f}% "
              f"{r['recall']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['roc_auc']:>7.3f} "
              f"{r['log_loss']:>8.4f} {r['cv_acc_mean']*100:>10.2f}% {cm:>14}")

    print("\nTop 8 NBA Stats Learned by Tuned Logistic Regression:")
    for idx in np.argsort(np.abs(tuned_lr_weights))[::-1][:8]:
        direction = "Home +" if tuned_lr_weights[idx] >= 0 else "Away +"
        print(f"  {feature_names[idx]:<26}: {tuned_lr_weights[idx]:+.4f}  ({direction})")

    with open("results/milestone4_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_final_plots(rows, feature_names, tuned_lr_weights)
    print("\nSaved milestone4_results.csv and 3 final charts inside plots/.")


if __name__ == "__main__":
    main()
