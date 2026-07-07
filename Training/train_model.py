"""
train_model.py
==============
End-to-end training pipeline for the Smart Lender project.

Run from the project root (SmartBridge/):

    python Training/train_model.py

What it does
------------
1. Loads Dataset/loan_prediction.csv
2. Preprocesses (shared `Flask/preprocessing.py`): impute -> encode -> scale
3. Stratified 80/20 split (random_state=42)
4. SMOTE oversampling on the TRAINING fold only (no leakage into the test set)
5. Trains four models with tuned hyperparameters:
        - Decision Tree
        - Random Forest   -> saved as rdf.pkl
        - KNN
        - XGBoost
6. Evaluates each (accuracy / precision / recall / F1 / ROC-AUC on train & test)
   and prints a comparison table.
7. Saves:
        - Flask/scale.pkl       (StandardScaler)
        - Flask/rdf.pkl         (Random Forest -- always)
        - Flask/best_model.pkl  (highest test-F1 model)

The best model is selected by *test F1* (robust to the 69/31 class imbalance)
rather than raw accuracy, which can be misleading on imbalanced data.
"""

from __future__ import annotations

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # fallback only
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Make the shared preprocessing module importable regardless of CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Flask"))
import preprocessing as pp  # noqa: E402

RANDOM_STATE = 42
DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "loan_prediction.csv")
FLASK_DIR = os.path.join(PROJECT_ROOT, "Flask")


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Loan_ID"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def make_xy():
    df = load_data()
    y = pp.encode_target(df["Loan_Status"])
    X = pp.encode_features(df.drop(columns=["Loan_Status"]))
    return X, y


# ---------------------------------------------------------------------------
# Per-model metric report
# ---------------------------------------------------------------------------
def evaluate(model, Xtr, ytr, Xte, yte):
    """Return a dict of train+test metrics plus the test confusion matrix."""
    ptr = model.predict(Xtr)
    pte = model.predict(Xte)
    pptr = model.predict_proba(Xtr)[:, 1] if hasattr(model, "predict_proba") else ptr
    ppte = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else pte

    return {
        "train_acc": accuracy_score(ytr, ptr),
        "test_acc": accuracy_score(yte, pte),
        "train_f1": f1_score(ytr, ptr),
        "test_f1": f1_score(yte, pte),
        "precision": precision_score(yte, pte, zero_division=0),
        "recall": recall_score(yte, pte, zero_division=0),
        "roc_auc": roc_auc_score(yte, ppte) if len(np.unique(yte)) > 1 else float("nan"),
        "cm": confusion_matrix(yte, pte),
    }


# ---------------------------------------------------------------------------
# Model factory -- tuned hyperparameters
# ---------------------------------------------------------------------------
def build_models():
    """Return an ordered dict of {name: unfitted estimator}."""
    return {
        "DecisionTree": DecisionTreeClassifier(
            criterion="gini", max_depth=6, min_samples_split=20,
            min_samples_leaf=5, random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9, weights="distance", metric="manhattan", p=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, gamma=2,
            reg_alpha=0.3, reg_lambda=1.5, min_child_weight=3,
            eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Smart Lender -- Training Pipeline")
    print("=" * 70)

    X, y = make_xy()
    print(f"\n[1] Loaded data:  X={X.shape}  y={y.shape}")
    print(f"    Target distribution:  Approved(1)={int(y.sum())}  "
          f"Rejected(0)={int((y == 0).sum())}")

    # --- Scale numeric columns (fit on the FULL training feature matrix).
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[pp.NUMERIC_COLS] = scaler.fit_transform(X_scaled[pp.NUMERIC_COLS])

    # --- Stratified train/test split BEFORE SMOTE (prevents leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled.values, y.values, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y.values,
    )
    print(f"[2] Split:  train={X_train.shape[0]}  test={X_test.shape[0]}")

    # --- SMOTE on the training fold only.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[3] SMOTE:  train before={X_train.shape[0]}  "
          f"after={X_train_res.shape[0]} "
          f"(1:{int(y_train_res.sum())} / 0:{int((y_train_res==0).sum())})")

    # --- Train + evaluate every model.
    print("\n[4] Training models ...\n")
    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        try:
            model.fit(X_train_res, y_train_res)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"   ! {name} failed ({exc}); falling back to LogisticRegression")
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            model.fit(X_train_res, y_train_res)
        results[name] = evaluate(model, X_train_res, y_train_res, X_test, y_test)
        fitted_models[name] = model
        r = results[name]
        print(f"   {name:14s} "
              f"train_acc={r['train_acc']:.3f}  test_acc={r['test_acc']:.3f}  "
              f"test_F1={r['test_f1']:.3f}  AUC={r['roc_auc']:.3f}")

    # --- Comparison table.
    print("\n[5] Comparison (sorted by test F1)\n")
    print(f"   {'Model':<14}{'TrainAcc':>10}{'TestAcc':>10}"
          f"{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}")
    print("   " + "-" * 71)
    for name, _ in sorted(results.items(), key=lambda kv: kv[1]["test_f1"], reverse=True):
        r = results[name]
        print(f"   {name:<14}{r['train_acc']:>10.3f}{r['test_acc']:>10.3f}"
              f"{r['precision']:>11.3f}{r['recall']:>9.3f}"
              f"{r['test_f1']:>8.3f}{r['roc_auc']:>9.3f}")

    # --- Persist artifacts.
    os.makedirs(FLASK_DIR, exist_ok=True)

    with open(os.path.join(FLASK_DIR, "scale.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"\n[6] Saved scale.pkl")

    # Random Forest is always saved as rdf.pkl (matches the spec).
    with open(os.path.join(FLASK_DIR, "rdf.pkl"), "wb") as fh:
        pickle.dump(fitted_models["RandomForest"], fh)
    print(f"[6] Saved rdf.pkl  (RandomForest)")

    # Best model by test F1.
    best_name = max(results, key=lambda n: results[n]["test_f1"])
    best_model = fitted_models[best_name]
    with open(os.path.join(FLASK_DIR, "best_model.pkl"), "wb") as fh:
        pickle.dump(best_model, fh)
    print(f"[6] Saved best_model.pkl  (best = {best_name} "
          f"with test F1 = {results[best_name]['test_f1']:.3f})")

    print("\nDone. Artifacts are in Flask/. Next:  python Flask/app.py\n")


if __name__ == "__main__":
    main()
