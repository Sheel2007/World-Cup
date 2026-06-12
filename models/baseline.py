"""XGBoost classical baseline — multiclass home/draw/away."""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT
from models.data_loader import load_features, split_by_date


MODEL_PATH = ROOT / "models" / "xgb.joblib"
SCALER_PATH = ROOT / "models" / "xgb_scaler.joblib"


def train():
    meta, X, y = load_features()
    X_tr, y_tr, _, X_te, y_te, _ = split_by_date(meta, X, y)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr_s, y_tr)
    acc = clf.score(X_te_s, y_te)
    print(f"XGBoost test accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved {MODEL_PATH.name}, {SCALER_PATH.name}")


def load():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def predict_proba(X: np.ndarray) -> np.ndarray:
    clf, scaler = load()
    return clf.predict_proba(scaler.transform(X))


if __name__ == "__main__":
    train()
