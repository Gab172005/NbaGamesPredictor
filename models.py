"""
Three beginner-friendly NBA game prediction models built from scratch with NumPy:
  1. KNN                - finds the k most similar past games
  2. GaussianNB         - compares average stats for home wins vs. home losses
  3. LogisticRegression - learns one weight per stat using simple gradient descent
"""

import math
import numpy as np


class KNN:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, k=15):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN just remembers the training games to compare against later
        self.X_train = X
        self.y_train = y
        return self

    def predict_proba(self, X):
        probs = []
        for game in X:
            # Straight-line (Euclidean) distance from this game to every training game
            distances = np.sqrt(np.sum((self.X_train - game) ** 2, axis=1))
            nearest_idx = np.argsort(distances)[:self.k]
            # Estimated home-win probability = fraction of the k neighbors won by home
            probs.append(np.mean(self.y_train[nearest_idx]))
        return np.array(probs, dtype=np.float64)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int64)


class GaussianNB:
    """Gaussian Naive Bayes classifier assuming bell-curve stats per class."""

    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        # Compute the average, variance, and overall frequency for losses (0) and wins (1)
        for c in (0, 1):
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + self.var_smoothing
            self.priors[c] = len(X_c) / len(y)
        return self

    def _log_score(self, X, c):
        # Sum log-probabilities across features (using logs prevents tiny underflow numbers)
        mean = self.means[c]
        var = self.vars[c]
        log_prior = math.log(self.priors[c])
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * math.pi * var) + ((X - mean) ** 2) / var,
            axis=1,
        )
        return log_prior + log_likelihood

    def predict_proba(self, X):
        log_p0 = self._log_score(X, 0)
        log_p1 = self._log_score(X, 1)
        # Convert the difference in log-scores into a 0-to-1 win probability
        diff = np.clip(log_p1 - log_p0, -500, 500)
        return 1.0 / (1.0 + np.exp(-diff))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int64)


class LogisticRegression:
    """Logistic Regression trained with simple batch gradient descent."""

    def __init__(self, lr=0.05, epochs=1000, l2=0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        # Squashes any score into a probability between 0 and 1
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        # Repeatedly nudge weights in the direction that lowers prediction error
        for _ in range(self.epochs):
            preds = self._sigmoid(X @ self.weights + self.bias)
            errors = preds - y
            # Optional L2 penalty gently shrinks large weights to avoid overfitting
            self.weights -= self.lr * ((X.T @ errors) / n_samples + self.l2 * self.weights)
            self.bias -= self.lr * np.mean(errors)
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

