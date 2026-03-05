import os
import joblib
import numpy as np
import tensorflow as tf

class HybridFraudDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.rf_model = None
        self.iso_model = None
        self.autoencoder = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        try:
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
            self.rf_model = joblib.load(os.path.join(self.models_dir, 'random_forest.pkl'))
            self.iso_model = joblib.load(os.path.join(self.models_dir, 'isolation_forest.pkl'))
            self.autoencoder = tf.keras.models.load_model(os.path.join(self.models_dir, 'autoencoder.keras'), compile=False)
            print("All models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}. Please ensure models exist and aren't corrupted.")

    def get_autoencoder_reconstruction_error(self, X_scaled):
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        return mse

    def predict(self, X):
        """
        Takes raw features X, scales them, and returns hybrid predictions.
        """
        # 1. Scale features
        X_scaled = self.scaler.transform(X)

        # 2. Random Forest prediction (Probability of fraud)
        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]

        # 3. Isolation Forest prediction
        # iso returns 1 for normal, -1 for anomaly
        iso_preds = self.iso_model.predict(X_scaled)
        # convert to 0 (normal) and 1 (anomaly)
        iso_scores = np.where(iso_preds == 1, 0, 1)

        # 4. Autoencoder reconstruction error
        ae_errors = self.get_autoencoder_reconstruction_error(X_scaled)
        
        # Determine dynamic threshold based on some simple heuristic (e.g. error > 10.0)
        # In a deep academic project, we might calculate a threshold at 95th percentile of normal data
        # For this hybrid model, we'll use a fixed threshold or combine it differently.
        # Let's use a threshold of 5.0 for MSE as an example, or use the normalized MSE as a feature
        ae_scores = np.where(ae_errors > 5.0, 1, 0)
        
        # Calculate Hybrid Score
        # Weighting: RF (0.6), Iso (0.2), AE (0.2)
        # RF outputs prob [0,1], Iso outputs 0/1, AE outputs 0/1 (based on threshold)
        hybrid_scores = (rf_probs * 0.6) + (iso_scores * 0.2) + (ae_scores * 0.2)
        
        # Final classification based on hybrid score
        final_predictions = np.where(hybrid_scores > 0.5, 1, 0)

        # Compile details into a dictionary for explainability/dashboard
        results = {
            'rf_probability': rf_probs,
            'iso_anomaly': iso_scores,
            'ae_reconstruction_error': ae_errors,
            'hybrid_score': hybrid_scores,
            'prediction': final_predictions
        }
        return results

    def get_scaler(self):
        return self.scaler

    def get_rf_model(self):
        return self.rf_model
