"""Compare QSVM vs XGBoost on held-out 2025 matches.

Metrics: accuracy, multinomial log-loss, Brier score (multiclass).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

sys.path.append(str(Path(__file__).resolve().parent))
from models.data_loader import load_features, split_by_date
from models import qsvm as qsvm_mod
from models import baseline as xgb_mod


def brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def evaluate(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=[0, 1, 2])
    br = brier_multiclass(y_true, probs)
    return {"model": name, "accuracy": acc, "log_loss": ll, "brier": br}


def main():
    meta, X, y = load_features()
    _, _, _, X_te, y_te, _ = split_by_date(meta, X, y)
    print(f"Eval set: {len(X_te)} matches in 2025 window")

    print("\nLoading XGBoost...")
    xgb_probs = xgb_mod.predict_proba(X_te)
    xgb_metrics = evaluate("XGBoost", y_te, xgb_probs)

    print("Loading QSVM (this computes the test kernel — slow)...")
    qsvm_probs = qsvm_mod.predict_proba(X_te)
    qsvm_metrics = evaluate("QSVM", y_te, qsvm_probs)

    print("\n=== Comparison ===")
    print(f"{'Model':<10} {'Accuracy':>10} {'LogLoss':>10} {'Brier':>10}")
    for m in (xgb_metrics, qsvm_metrics):
        print(f"{m['model']:<10} {m['accuracy']:>10.4f} {m['log_loss']:>10.4f} {m['brier']:>10.4f}")

    diff = qsvm_metrics["accuracy"] - xgb_metrics["accuracy"]
    print(f"\nΔ accuracy (QSVM - XGBoost): {diff:+.4f}")
    target = 0.05
    status = "WITHIN" if abs(diff) <= target else "OUTSIDE"
    print(f"PRD target: |Δ| ≤ {target} — {status} target.")


if __name__ == "__main__":
    main()
