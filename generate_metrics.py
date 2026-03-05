import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score

def generate_metrics():
    print("Loading data for quick evaluation...")
    df = pd.read_csv('data/creditcard.csv')
    
    # Simple split that matches train.py
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Loading scaler and model...")
    scaler = joblib.load('models/scaler.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    
    X_test_scaled = scaler.transform(X_test)
    
    print("Predicting...")
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Saving metrics...")
    acc = float(accuracy_score(y_test, rf_preds))
    metrics = {
        "accuracy": acc,
        "precision": float(precision_score(y_test, rf_preds)),
        "recall": float(recall_score(y_test, rf_preds)),
        "f1": float(f1_score(y_test, rf_preds)),
        "roc_auc": float(roc_auc_score(y_test, rf_probs)),
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("metrics.json successfully created.")

if __name__ == "__main__":
    generate_metrics()
