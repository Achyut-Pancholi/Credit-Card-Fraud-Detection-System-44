# Credit Card Fraud Detection using Hybrid Machine Learning & Anomaly Detection

## Project Overview
This project is an intelligent fraud detection system designed to identify suspicious credit card transactions quickly and accurately. Traditional fraud detection systems often struggle with highly imbalanced datasets and failing to detect new or unusual fraud patterns. This project addresses these limitations by employing a hybrid machine learning approach.

## Problem Statement
Online transactions are increasing rapidly, and credit card fraud is also rising. The challenges include detecting fraud quickly enough, handling transactions where fraudulent cases are an extreme minority, and identifying unknown fraud patterns. This project aims to solve these problems.

## Objectives
- Identify suspicious transactions using machine learning.
- Handle imbalanced fraud data effectively.
- Improve fraud detection accuracy.
- Provide a simple real-time prediction dashboard.
- Help reduce financial loss and improve transaction security.

## Methodology
The workflow consists of:
1. **Data Loading:** Loading transaction data.
2. **Data Preprocessing:** Checking missing values, separating features/target, scaling via `StandardScaler`, and applying SMOTE for class imbalance.
3. **Model Training:** Training Random Forest, Isolation Forest, and an Autoencoder Neural Network.
4. **Hybrid Prediction:** Combining the model predictions into a final fraud score.

## Hybrid Model Explanation
The core innovation of this system is the integration of multiple models:
1. **Random Forest:** A classification model optimized to distinguish normal and fraudulent transactions.
2. **Isolation Forest:** An anomaly detection model that identifies unusual and potentially novel fraudulent transaction behavior.
3. **Autoencoder Neural Network:** A deep learning model trained to reconstruct normal transactions. High reconstruction errors indicate anomalies.

These predictions are combined to form a holistic decision, capturing both known and unknown fraud patterns.

## Explainable AI with SHAP
The project incorporates SHAP (SHapley Additive Explanations) to provide transparency to model predictions. For each prediction, the system displays the feature importance and the top contributing factors that led to the transaction being flagged as fraud or not.

## Streamlit Dashboard
A real-time Streamlit dashboard allows users to upload a transaction dataset or test single transactions. It displays the predicted outcomes, probability scores, project metrics, confusion matrix, and feature importance charts.

## Model Evaluation
The models are evaluated on metrics including Accuracy, Precision, Recall, F1 Score, and ROC-AUC to ensure robust performance across both majority (normal) and minority (fraud) classes.

## Project Structure
```
credit_card_fraud_project/
│
├── data/
│   └── creditcard.csv          # Raw dataset (must be added manually)
│
├── models/                     # Auto-generated directory after training
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   ├── autoencoder.keras
│   └── metrics.json
│
├── train.py                    # Model training & persistence script
├── model.py                    # Hybrid inference engine logic
├── app.py                      # Streamlit real-time dashboard
├── generate_metrics.py         # Utility to explicitly evaluate models
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
└── changes.md                  # Project version tracker
```

## How to Run the Project

### 1. Setup Environment
Ensure you have Python 3.10 to 3.12 installed.
Open a terminal in the `credit_card_fraud_project` directory.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your `creditcard.csv` dataset is placed inside the `data/` folder.

### 3. Train Models
Run the training pipeline. This script scales features, applies SMOTE, trains all three models (Random Forest, Isolation Forest, and Autoencoder), generates evaluation metrics, and saves everything to the `models/` directory.

```bash
python train.py
```
*(Note: Autoencoder training can take 5-10 minutes depending on your hardware).*

### 4. Launch Dashboard
Start the real-time prediction dashboard using Streamlit.

```bash
streamlit run app.py
```
This will open the web application in your default browser at `http://localhost:8501`.

### 5. How to Use and Test the Dashboard

Once the Streamlit server is running (`http://localhost:8501`), follow these steps to test the system:

#### Testing the Interactive Simulator
1. In the left navigation menu, click **Interactive Simulator**.
2. This page allows you to manually tweak parameters using sliders.
3. Select a **Quick Load Scenario Preset** such as "High-Risk Fraud Profile". This will instantly load values known to trigger the fraud detection models.
4. Click **Run Analysis on Scenario**.
5. The system will display the Hybrid Risk Score and a Plain English AI Explanation powered by SHAP detailing exactly how each feature influenced the Random Forest's decision.

#### Testing Real-Time Single Transactions (Synthetic Data)
1. In the left navigation menu, click **Fraud Prediction (Batch/Sample)**.
2. Scroll down to the bottom section: *"Or test a single sample transaction"*.
3. Click the **Generate & Test Random Sample** button.
4. The system will magically generate a 30-feature transaction array (representing V1-V28, Amount, Time).
5. Watch the progress bars as the system runs the transaction through the Random Forest, Isolation Forest, and Deep Learning Autoencoder.
6. A banner will appear classifying it as either **✅ Safe** or **🚨 FRAUD**, along with the hybrid combination score.
7. Scroll down to view the **Model Breakdown** and the **SHAP Feature Importance Chart**, which visually explains exactly *why* the models made that decision based on the specific fields.

#### Testing Batch Inference (CSV Upload)
If you want to test the bulk prediction engine, you can create a mock CSV file to upload.
1. Create a new file on your computer named `mock_test_data.csv`.
2. Copy and paste the following mock transaction data into the file (it contains headers for Time, V1-V28, and Amount):

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
0.0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62
0.0,1.19185711131486,0.266150712059638,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137286,0.489095015896574,-0.143772296441519,0.635558093294411,0.463917041022171,-0.114804663116819,-0.183361270123985,-0.145783041416806,-0.0690831351230232,-0.225775248039822,-0.638671952771851,0.101288021079549,-0.339846475510427,0.167170404423458,0.12589453229653,-0.0089830991432281,0.0147241691924927,2.69
```

3. In the left navigation menu of the dashboard, click **Fraud Prediction**.
4. Drag and drop your `mock_test_data.csv` file into the upload box at the top.
5. The dashboard will preview your data. Click the **Run Batch Prediction** button entirely.
6. The system will evaluate both rows against all three models simultaneously and append the results.
7. You can then view the results directly in the UI or click the **📥 Download Results CSV** button to download the finalized classification dataset!

#### Viewing Model Accuracy
1. Click **Model Results** in the left sidebar menu.
2. Here, you can visually verify the system's performance on the historic test-split dataset, checking the True accuracy, precision, and recall alongside the Confusion Matrix heat map.

## Future Scope
- Integration with real streaming data platforms like Apache Kafka.
- Model retraining pipelines for continuous learning from new fraud cases.
- Expanding explainability to natural language descriptions.
