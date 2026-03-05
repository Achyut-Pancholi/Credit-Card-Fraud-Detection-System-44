# Changes Tracker

### Version 1.0 (Current)
- Project initialization and directory structure set up.
- Project documentation established (`README.md`, `changes.md`).
- Virtual environment and dependencies initialized.
- Dataset (creditcard.csv) correctly placed in the `data/` folder.

### Version 1.1
- Feature scaling (`StandardScaler`) and extreme class imbalance handling (`SMOTE`) implemented in `train.py`.
- Random Forest classification model integrated and trained with optimized estimators.
- Isolation Forest anomaly detection model integrated for unsupervised outlier detection.

### Version 1.2
- Deep Learning Autoencoder model integrated using `Keras` (`.keras` format) for high-end pattern reconstruction.
- Hybrid prediction logical engine (`model.py`) deployed to calculate dynamic, weighted fraud scores.

### Version 1.3
- SHAP (SHapley Additive Explanations) integrated directly against the classification components.
- SHAP visualization optimized using native Matplotlib charts to prevent streaming interface freezes.
- Fully interactive `Streamlit` dashboard (`app.py`) built with Batch CSV Prediction and Synthetic Real-Time data generation.
- Real dynamic model accuracy metrics integrated via `metrics.json`.
- UI upgraded to natively support both Light and Dark mode.

### Version 1.4 (Final Polishing)
- **Interactive Simulator** deployed allowing users to manually tweak standard and extreme V-features bounds.
- Added natural-language **AI Explanations** directly tied to the specific model (Random Forest) that SHAP is explaining.
- Upgraded Model Results view to utilize dynamic **Plotly Gauge Charts** and re-established the Global Feature Importance visualization.
- Implemented **Intelligent Batch Dataframe viewing** in Streamlit, ensuring predicted Fraud rows dynamically bubble to the top of the UI to bypass 200MB memory crash limits.
- **Data Dictionary Tooltips** integrated directly into the dashboard for user education.
