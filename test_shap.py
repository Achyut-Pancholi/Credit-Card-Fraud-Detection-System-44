import joblib

rf = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

import numpy as np
feature_row = np.zeros((1, 30))
# Let's try more extreme values to guarantee RF predicts Fraud
feature_row[0, 14] = -15.0 # V14
feature_row[0, 17] = -12.0 # V17
feature_row[0, 12] = -14.0 # V12
feature_row[0, 10] = -10.0 # V10
feature_row[0, 4] = 12.0 # V4
feature_row[0, 3] = -10.0 # V3
feature_row[0, 29] = 999.0

scaled = scaler.transform(feature_row)
print("RF Probabilities:", rf.predict_proba(scaled))
