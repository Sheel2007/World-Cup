"""Quantum SVM match predictor.

Pipeline: StandardScaler -> PCA(6) -> ZZFeatureMap -> FidelityQuantumKernel -> QSVC.
Three one-vs-rest QSVCs combined with softmax-normalised decision scores → 3-class probs.

Quantum kernel computation is O(N^2) and slow on a simulator; we subsample the training set
to keep runtime tractable. This is standard practice for QSVM demos.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT, QSVM_PCA_COMPONENTS, QSVM_FEATURE_MAP_REPS
from models.data_loader import load_features, split_by_date


MODEL_PATH = ROOT / "models" / "qsvm.joblib"
SCALER_PATH = ROOT / "models" / "qsvm_scaler.joblib"
PCA_PATH = ROOT / "models" / "qsvm_pca.joblib"
ANGLE_PATH = ROOT / "models" / "qsvm_angle_scaler.joblib"
KERNEL_PATH = ROOT / "models" / "qsvm_kernel.joblib"

DEFAULT_TRAIN_CAP = 400  # subsample to keep kernel matrix tractable on simulator


def _build_quantum_kernel(n_features: int):
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fm = ZZFeatureMap(
        feature_dimension=n_features,
        reps=QSVM_FEATURE_MAP_REPS,
        entanglement="linear",
    )
    return FidelityQuantumKernel(feature_map=fm)


def _subsample(X: np.ndarray, y: np.ndarray, cap: int, rng):
    if len(X) <= cap:
        return X, y
    # stratified subsample
    idx_per_class = [np.where(y == c)[0] for c in np.unique(y)]
    per_class_cap = cap // len(idx_per_class)
    picks = []
    for idx in idx_per_class:
        if len(idx) <= per_class_cap:
            picks.append(idx)
        else:
            picks.append(rng.choice(idx, size=per_class_cap, replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return X[sel], y[sel]


def train(train_cap: int = DEFAULT_TRAIN_CAP, seed: int = 42):
    meta, X, y = load_features()
    X_tr_all, y_tr_all, _, X_te, y_te, _ = split_by_date(meta, X, y)
    print(f"Train pool: {len(X_tr_all)}  Test: {len(X_te)}")

    scaler = StandardScaler().fit(X_tr_all)
    X_tr_s = scaler.transform(X_tr_all)
    X_te_s = scaler.transform(X_te)

    pca = PCA(n_components=QSVM_PCA_COMPONENTS, random_state=seed).fit(X_tr_s)
    X_tr_p = pca.transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)

    # ZZFeatureMap treats inputs as rotation angles — bound to [0, π] for stable kernels.
    angle = MinMaxScaler(feature_range=(0.0, float(np.pi))).fit(X_tr_p)
    X_tr_a = angle.transform(X_tr_p)
    X_te_a = np.clip(angle.transform(X_te_p), 0.0, float(np.pi))

    rng = np.random.default_rng(seed)
    X_tr_sub, y_tr_sub = _subsample(X_tr_a, y_tr_all, train_cap, rng)
    print(f"Quantum train subsample: {len(X_tr_sub)} (cap={train_cap})")

    print(f"Building quantum kernel ({QSVM_PCA_COMPONENTS} qubits, ZZFeatureMap reps={QSVM_FEATURE_MAP_REPS})")
    kernel = _build_quantum_kernel(QSVM_PCA_COMPONENTS)

    print("Computing training kernel matrix...")
    K_train = kernel.evaluate(x_vec=X_tr_sub)
    print(f"Train kernel shape: {K_train.shape}")

    clf = SVC(kernel="precomputed", probability=True, random_state=seed, C=2.0)
    clf.fit(K_train, y_tr_sub)

    print("Computing test kernel matrix...")
    K_test = kernel.evaluate(x_vec=X_te_a, y_vec=X_tr_sub)
    acc = clf.score(K_test, y_te)
    print(f"QSVM test accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "X_train": X_tr_sub, "y_train": y_tr_sub}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(angle, ANGLE_PATH)


def load():
    bundle = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    angle = joblib.load(ANGLE_PATH)
    return bundle, scaler, pca, angle


def predict_proba(X: np.ndarray) -> np.ndarray:
    bundle, scaler, pca, angle = load()
    X_a = np.clip(angle.transform(pca.transform(scaler.transform(X))), 0.0, float(np.pi))
    kernel = _build_quantum_kernel(QSVM_PCA_COMPONENTS)
    K = kernel.evaluate(x_vec=X_a, y_vec=bundle["X_train"])
    return bundle["clf"].predict_proba(K)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cap", type=int, default=DEFAULT_TRAIN_CAP,
                   help="Max training samples for quantum kernel")
    args = p.parse_args()
    train(train_cap=args.cap)
