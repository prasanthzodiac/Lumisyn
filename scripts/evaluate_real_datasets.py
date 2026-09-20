import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

datasets = [
    ("wisconsin", "lumisyn/data/kaggle_datasets/wisconsin_clean.csv", "target"),
    ("diabetes", "lumisyn/data/kaggle_datasets/pima_diabetes.csv", "Outcome"),
    ("adult", "lumisyn/data/kaggle_datasets/adult.csv", "class"),
    ("cardio", "lumisyn/data/kaggle_datasets/cardio_train.csv", "cardio"),
]

print("================================================================================")
print(" VERIFYING REAL PUBLIC TABULAR DATASETS (TRTR AUROC & AUPRC)")
print("================================================================================")

for name, path, target_col in datasets:
    if not os.path.exists(path):
        print(f"FAILED TO FIND: {path}")
        continue
    df = pd.read_csv(path)
    print(f"\n--- {name.upper()} ---")
    print(f"File: {path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check deduplication / patient id
    if "id" in df.columns:
        n_unique = df["id"].nunique()
        print(f"ID Column detected: {n_unique} unique IDs / {len(df)} rows.")
        if n_unique != len(df):
            print(f"Deduplicating to 1 row per patient ID...")
            df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
            print(f"Deduplicated shape: {df.shape}")
        else:
            print("Verified: Exactly 1 row per patient ID.")
    else:
        print("No explicit patient ID column; row-level records.")
        
    y = df[target_col]
    if y.dtype == object:
        y = (y.astype(str).str.strip().isin([">50K", ">50K.", "1", "True"])).astype(int)
    else:
        y = y.astype(int)
        
    X = df.drop(columns=[target_col])
    if "id" in X.columns:
        X = X.drop(columns=["id"])
        
    # One-hot encode categoricals for evaluation
    X_enc = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    pos_rate = float(y.mean())
    print(f"Target Column: '{target_col}' | Positive prevalence: {pos_rate:.4f} ({pos_rate:.2%})")
    print(f"TRTR AUROC: {auroc:.4f} (Status: {'RETAIN (>0.5)' if auroc > 0.6 else 'DROP NEAR 0.5'})")
    print(f"TRTR AUPRC: {auprc:.4f} (Baseline prevalence: {pos_rate:.4f})")

print("\n================================================================================")
