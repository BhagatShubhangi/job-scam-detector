import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

from features import FEATURE_COLS

def train():
    # Load processed data
    df = pd.read_csv('../data/processed/features.csv')

    X = df[FEATURE_COLS].fillna(0)
    y = df['fraudulent']

    # Train/test split — stratified to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Fraud in test set: {y_test.sum()}")

    # SMOTE — oversample fraud cases in training only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE — Train size: {X_res.shape[0]}")
    print(f"Fraud cases in training: {y_res.sum()}")

    # --- Baseline: Logistic Regression ---
    print("\n=== Logistic Regression (Baseline) ===")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_res, y_res)
    print(classification_report(y_test, lr.predict(X_test)))

    # --- Main Model: XGBoost ---
    print("\n=== XGBoost ===")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=10,
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_res, y_res)
    print(classification_report(y_test, xgb.predict(X_test)))

    # Confusion matrix
    cm = confusion_matrix(y_test, xgb.predict(X_test))
    print("Confusion Matrix:")
    print(f"  True Negatives  (real, called real): {cm[0][0]}")
    print(f"  False Positives (real, called fake): {cm[0][1]}")
    print(f"  False Negatives (fake, called real): {cm[1][0]}")
    print(f"  True Positives  (fake, called fake): {cm[1][1]}")

    # Save model + feature list
    os.makedirs('../models', exist_ok=True)
    joblib.dump(xgb, '../models/xgb_model.pkl')
    joblib.dump(FEATURE_COLS, '../models/feature_cols.pkl')
    print("\nModel saved to models/xgb_model.pkl")

if __name__ == '__main__':
    train()