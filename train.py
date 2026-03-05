import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import json

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, X_train_scaled, y_train

def build_and_train_autoencoder(X_train_normal):
    print("Building Autoencoder...")
    input_dim = X_train_normal.shape[1]
    
    # Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation="relu")(input_layer)
    encoder = Dense(16, activation="relu")(encoder)
    encoder = Dense(8, activation="relu")(encoder)
    
    decoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(32, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Training Autoencoder...")
    # Train only on normal transactions
    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        verbose=1
    )
    
    autoencoder.save('models/autoencoder.keras')
    return autoencoder

def train_models():
    data_path = 'data/creditcard.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    X_train, X_test, y_train_resampled, y_test, X_train_scaled_original, y_train_orig = load_and_preprocess_data(data_path)
    
    # 1. Train Random Forest (Classification)
    print("Training Random Forest...")
    # Using limited estimators for faster training in academic context
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    rf_model.fit(X_train, y_train_resampled)
    joblib.dump(rf_model, 'models/random_forest.pkl')
    
    # Evaluate RF
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, rf_preds))
    roc_auc = float(roc_auc_score(y_test, rf_probs))
    
    print("\nRandom Forest Performance:")
    print("Accuracy:", acc)
    print("ROC-AUC:", roc_auc)
    print(classification_report(y_test, rf_preds))
    
    # 2. Train Isolation Forest (Anomaly Detection)
    # Train it on the original scaled data (not SMOTE) to learn anomalies
    print("Training Isolation Forest...")
    iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso_model.fit(X_train_scaled_original)
    joblib.dump(iso_model, 'models/isolation_forest.pkl')
    
    # Evaluate Iso Forest
    # IsoForest returns 1 for normal, -1 for anomaly. Map to 0 (normal), 1 (fraud)
    iso_preds = iso_model.predict(X_test)
    iso_preds = np.where(iso_preds == 1, 0, 1)
    print("\nIsolation Forest Performance:")
    print(classification_report(y_test, iso_preds))
    
    # 3. Train Autoencoder
    # Train only on majority class (normal transactions = 0)
    train_normal_indices = np.where(y_train_orig == 0)[0]
    X_train_normal = X_train_scaled_original[train_normal_indices]
    
    build_and_train_autoencoder(X_train_normal)
    
    # Evaluate hybrid approach on test set to capture true system metrics
    # To do this accurately, we load the HybridFraudDetector (or mock its logic)
    # We will use the RF metrics as a baseline for the dashboard for simplicity
    # but let's calculate the precision, recall and confusion matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        "accuracy": acc,
        "precision": float(precision_score(y_test, rf_preds)),
        "recall": float(recall_score(y_test, rf_preds)),
        "f1": float(f1_score(y_test, rf_preds)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("\nAll models trained and saved successfully in 'models/' directory.")

if __name__ == "__main__":
    train_models()
