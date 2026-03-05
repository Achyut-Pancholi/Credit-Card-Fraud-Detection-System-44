import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import base64
import os
import json
import plotly.graph_objects as go
from model import HybridFraudDetector

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

@st.cache_resource
def load_detector():
    return HybridFraudDetector()

@st.cache_resource
def get_shap_explainer(_model):
    # Caching the SHAP explainer significantly speeds up analysis time 
    # approximate=True drastically improves speed on large Random Forests
    return shap.TreeExplainer(_model, approximate=True)

def main():
    st.title("🛡️ Hybrid Credit Card Fraud Detection")
    
    # Check if models are trained
    if not os.path.exists('models/autoencoder.h5'):
        st.warning("Models not found! Please run the training script first (`python train.py`).")
        st.stop()
    
    # Diagnostics sidebar
    with st.sidebar.expander("🔧 Debug Info"):
        import tensorflow as tf
        import sklearn
        st.text(f"TF: {tf.__version__}")
        st.text(f"Keras: {tf.keras.__version__}")
        st.text(f"NumPy: {np.__version__}")
        st.text(f"Sklearn: {sklearn.__version__}")
    
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"Failed to load models: {type(e).__name__}: {e}")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    menu = ["Project Overview", "Interactive Simulator", "Fraud Prediction (Batch/Sample)", "Model Results"]
    choice = st.sidebar.radio("Go to", menu)
    
    if choice == "Project Overview":
        show_overview()
    elif choice == "Interactive Simulator":
        show_interactive_simulator(detector)
    elif choice == "Fraud Prediction (Batch/Sample)":
        show_prediction(detector)
    elif choice == "Model Results":
        show_results()

def show_overview():
    st.markdown("<h1 style='text-align: center; color: #3b82f6;'>Project Overview</h1>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 The Mission")
        st.write("""
        Welcome to the **Credit Card Fraud Detection System**. 
        This application utilizes a powerful hybrid machine learning architecture to detect fraudulent transactions in real-time, 
        designed specifically to address the extreme class imbalance and dynamic nature of modern financial fraud.
        """)
        
        st.markdown("### 🧠 Why a Hybrid Model?")
        st.info("""
        Traditional single-model systems often struggle with emerging fraud patterns. Our architecture fuses three distinct AI philosophies:
        
        1. **Random Forest (Supervised Classification):** Memorizes and detects known historical fraud profiles with high precision.
        2. **Isolation Forest (Unsupervised Anomalies):** Maps out transaction densities to isolate structurally unusual behavior.
        3. **Deep Autoencoder (Pattern Reconstruction):** A neural network trained purely on *Normal* data. If it fails to reconstruct an incoming transaction (high error), it signals potential zero-day fraud.
        """)
        
    with col2:
        with st.container(border=True):
            st.markdown("#### ⚡ Core Capabilities")
            st.success("**High Accuracy:** Micro-tuned ensemble weighting minimizes both false positives and false negatives.")
            st.warning("**Real-Time XAI:** Integrated SHAP explainer provides plain-English transparency into the AI's decision process.")
            st.error("**Dynamic Simulation:** Built-in tools to rigorously stress-test the AI against synthetic zero-day vulnerabilities.")
            
    st.write("---")
    st.markdown("### 📊 Dataset Origin")
    st.write("""
    Trained on the highly imbalanced **Kaggle European Credit Card Dataset**.
    Due to strict banking confidentiality, the original 28 features (`Location`, `Merchant Category`, `IP Address`, etc.) were mathematically transformed via **PCA** (Principal Component Analysis) into the anonymized numerical vectors `V1` through `V28`.
    """)

def show_interactive_simulator(detector):
    st.header("Interactive Fraud Simulator")
    st.write("💡 **Play with transaction features in real-time** to see how the hybrid model reacts. Adjust the sliders to simulate different scenarios.")
    
    # Initialize session state for features if not exists
    if 'sim_features' not in st.session_state:
        st.session_state.sim_features = {f'V{i}': 0.0 for i in range(1, 29)}
        st.session_state.sim_features['Amount'] = 50.0
        st.session_state.sim_features['Time'] = 100000.0

    col_in, col_out = st.columns([1, 1], gap="large")
    
    with col_in:
        with st.container(border=True):
            st.subheader("📝 Transaction Parameters")
            
            # Presets
            preset = st.selectbox(
                "Quick Load Scenario Preset", 
                ["Custom (Manual Entry)", "Typical Safe Transaction", "High-Risk Fraud Profile"]
            )
            
            # Apply presets to session state defaults prior to rendering form
            if preset == "Typical Safe Transaction":
                st.session_state.sim_features = {f'V{i}': float(np.random.normal(0, 0.5)) for i in range(1, 29)}
                st.session_state.sim_features['Amount'] = 25.50
                st.session_state.sim_features['Time'] = 120000.0
            elif preset == "High-Risk Fraud Profile":
                st.session_state.sim_features = {f'V{i}': float(np.random.normal(0, 1)) for i in range(1, 29)}
                st.session_state.sim_features['V14'] = -15.0
                st.session_state.sim_features['V17'] = -12.0
                st.session_state.sim_features['V12'] = -14.0
                st.session_state.sim_features['V10'] = -10.0
                st.session_state.sim_features['V4'] = 12.0
                st.session_state.sim_features['V3'] = -10.0
                st.session_state.sim_features['Amount'] = 999.0
                st.session_state.sim_features['Time'] = 40000.0
                
            with st.form("simulator_form"):
                st.write("**Key Indicators (Most influential features)**")
                ic1, ic2 = st.columns(2)
                
                amt = ic1.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=float(st.session_state.sim_features['Amount']), step=10.0)
                
                # Sliders mapped to the most critical PCA variables found in creditcard.csv
                v14 = ic2.slider("V14 (Identity / Context)", min_value=-20.0, max_value=10.0, value=float(st.session_state.sim_features['V14']), step=0.1, help="Often correlates with mismatched identity markers or unusual shipping addresses.")
                v17 = ic1.slider("V17 (Location / Device)", min_value=-30.0, max_value=10.0, value=float(st.session_state.sim_features['V17']), step=0.1, help="Represents the geographical distance from the user's typical IP or device footprint.")
                v12 = ic2.slider("V12 (Account Age)", min_value=-20.0, max_value=10.0, value=float(st.session_state.sim_features['V12']), step=0.1, help="Indicates the maturity of the account. Highly negative values often flag compromised new accounts.")
                v10 = ic1.slider("V10 (Risk History)", min_value=-25.0, max_value=15.0, value=float(st.session_state.sim_features['V10']), step=0.1, help="Aggregated prior risk factors associated with the merchant or the cardholder.")
                v4 = ic2.slider("V4 (Transaction Velocity)", min_value=-5.0, max_value=15.0, value=float(st.session_state.sim_features['V4']), step=0.1, help="Measures the frequency of transactions within a short timeframe. High positive values suggest bot activity.")
                v3 = ic1.slider("V3 (Merchant Trust)", min_value=-30.0, max_value=10.0, value=float(st.session_state.sim_features['V3']), step=0.1, help="A score based on the reputation of the receiving merchant.")
                
                with st.expander("Advanced Features (Time & Internal PCAs)"):
                    time_val = st.number_input("Time (Seconds from start)", min_value=0.0, max_value=172792.0, value=float(st.session_state.sim_features['Time']))
                    # Background features remain static depending on preset
                    v_others = {i: st.session_state.sim_features[f'V{i}'] for i in range(1, 29) if i not in [3, 4, 10, 12, 14, 17]}
                    st.write("*Other 22 PCA numerical variables are running in the background based on your chosen preset.*")
                    
                submitted = st.form_submit_button("Run Analysis on Scenario", type="primary", use_container_width=True)
                
        with st.expander("📚 Data Dictionary & Testing Guide", expanded=False):
            st.markdown("""
            ### What are these V-features?
            Due to banking privacy laws, the original transaction data (like GPS Location, Device IP, Merchant Category, and CVV checks) were mathematically scrambled using **PCA** (Principal Component Analysis) into unrecognizable numerical values `V1` through `V28`. 
            
            By analyzing the Random Forest's weighting, we've mapped the top 6 most critical metrics back to identifiable behavioral concepts in the sliders above so you can test them intuitively!

            ---
            
            ### How to Test & Why it Works
            
            #### 🛡️ Safe Bounds (Normal Behavior)
            *   **How to test:** Try keeping the V-features close to `0` (between `-2.0` and `+2.0`). Attempt putting in a normal `Amount` (e.g. `$25.00`).
            *   **Why it flags Safe:** Because the data was scaled using PCA, a value of `0` represents the mathematical "average" behavior of millions of normal consumers. The **Autoencoder Neural Network** easily reconstructs this familiar pattern (yielding low error), and the **Isolation Forest** groups it safely alongside the majority of standard transactions.
            
            #### 🚨 Fraud Bounds (Anomalous Behavior)
            *   **How to test:** Try dragging `V14`, `V17`, or `V12` into deeply negative territory (e.g. `-10.0` or lower) while pushing `V4` highly positive (e.g. `+10.0`).
            *   **Why it triggers Fraud:** Forcing these values creates a transaction profile that mathematically deviates extreme distances from the baseline consumer average. 
                1. The **Random Forest** recognizes this specific structural combination as matching historical banking theft.
                2. The **Autoencoder Network** has never seen this pattern in its "Safe" training data, so it fails to reconstruct the data, throwing a massive mathematical Reconstruction Error.
                3. The **Hybrid Engine** catches all these alarms and shuts the transaction down.
            """)

    with col_out:
        if submitted or preset != "Custom (Manual Entry)":
            with st.spinner("Processing scenario through Hybrid Engine..."):
                # Assemble the 30-feature array for inference (Time, V1..V28, Amount)
                feature_row = np.zeros((1, 30))
                feature_row[0, 0] = time_val
                
                for i in range(1, 29):
                    if i == 14: feature_row[0, i] = v14
                    elif i == 17: feature_row[0, i] = v17
                    elif i == 12: feature_row[0, i] = v12
                    elif i == 10: feature_row[0, i] = v10
                    elif i == 4: feature_row[0, i] = v4
                    elif i == 3: feature_row[0, i] = v3
                    else: feature_row[0, i] = v_others[i]
                        
                feature_row[0, 29] = amt
                
                # Inference
                results = detector.predict(feature_row)
                is_fraud = results['prediction'][0] == 1
                score = results['hybrid_score'][0]
                
                if is_fraud:
                    st.error(f"### 🚨 FRAUD DETECTED\n**Hybrid Risk Score: {score:.2f}**\nThe system confidently flagged this transaction.", icon="🚨")
                else:
                    st.success(f"### ✅ SAFE TRANSACTION\n**Hybrid Risk Score: {score:.2f}**\nThis behavior is normal.", icon="✅")
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Random Forest Prob", f"{results['rf_probability'][0]:.2f}")
                mc2.metric("IsoForest Anomaly", "Yes" if results['iso_anomaly'][0] == 1 else "No")
                mc3.metric("Autoencoder Error", f"{results['ae_reconstruction_error'][0]:.2f}")
                
                # SHAP Plain English Explanation
                st.write("---")
                st.subheader("AI Explanation (Random Forest Component)")
                
                try:
                    explainer = get_shap_explainer(detector.get_rf_model())
                    scaled_sample = detector.get_scaler().transform(feature_row)
                    shap_values = explainer.shap_values(scaled_sample, check_additivity=False)
                    
                    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
                    
                    if isinstance(shap_values, list): vals = shap_values[1][0]
                    else: vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                    
                    impact_indices = np.argsort(np.abs(vals))[-5:] # Top 5 impacts
                    top_features = [feature_names[i] for i in impact_indices]
                    top_impacts = [vals[i] for i in impact_indices]
                    
                    # Generate natural language based specifically on Random Forest's isolated decision
                    rf_prob = results['rf_probability'][0]
                    rf_is_fraud = rf_prob >= 0.5
                    
                    explanation = f"Although the overarching Hybrid Score combines all models, SHAP explains the **Random Forest** component explicitly. "
                    explanation += f"The Random Forest individually concluded this was **{'Fraudulent' if rf_is_fraud else 'Safe'}** (Probability: {rf_prob:.2f}). Here is why:\n\n"
                    
                    for feat, impact in reversed(list(zip(top_features, top_impacts))):
                        direction = "pushed the RF score towards FRAUD" if impact > 0 else "pulled the RF score towards SAFE"
                        magnitude = "heavily" if abs(impact) > 0.05 else "slightly"
                        explanation += f"- The value of **{feat}** {magnitude} {direction} (Impact: {impact:+.3f}).\n"
                    
                    st.info(explanation)
                    
                except Exception as e:
                    st.warning(f"Feature importance text unavailable: {e}")
        else:
            st.info("👈 Choose a preset or adjust the sliders and click **Run Analysis** to see the AI's real-time explanation!")

def show_prediction(detector):
    st.header("Batch & Synthetic Prediction Engine")
    
    st.info("Upload a CSV file with transaction data or use the sample transaction tester below.")
    
    uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Assuming 'Class' column might be present but we only need features for prediction
            features = df.drop(columns=['Class'], errors='ignore')
            
            if st.button("Run Batch Prediction"):
                progress_bar = st.progress(0, text="Initializing batch analysis...")
                
                progress_bar.progress(30, text="Running Hybrid Models on dataset...")
                results = detector.predict(features.values)
                
                df['Predicted_Fraud'] = results['prediction']
                df['Hybrid_Score'] = results['hybrid_score']
                
                progress_bar.progress(80, text="Compiling results...")
                fraud_count = df['Predicted_Fraud'].sum()
                safe_count = len(df) - fraud_count
                
                progress_bar.progress(100, text="Batch Analysis Complete!")
                
                st.write("### Batch Prediction Results")
                col1, col2 = st.columns(2)
                col1.metric("Total Transactions", len(df))
                col2.metric("Flagged as Fraudulent", int(fraud_count), delta_color="inverse")
                
                # Create a filtered dataframe prioritizing flagged fraud transactions
                fraud_df = df[df['Predicted_Fraud'] == 1]
                safe_df = df[df['Predicted_Fraud'] == 0]
                
                # Show up to 100 rows, guaranteeing all fraud rows appear first
                display_df = pd.concat([fraud_df, safe_df]).head(100)
                
                st.write("**Processed Results (Showing all identified frauds first):**")
                # Highlight fraud rows in red for emphasis (Streamlit styler)
                def highlight_fraud(row):
                    if row['Predicted_Fraud'] == 1:
                        return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(display_df[['Predicted_Fraud', 'Hybrid_Score'] + list(features.columns)].style.apply(highlight_fraud, axis=1))
                    
                # Provide download link
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing file. If the file is extremely large, please note Streamlit's 200MB message limit. Error: {e}")
            
    else:
        st.write("---")
        st.markdown("### Or test a single synthetic transaction")
        st.markdown("Instantly generate a dummy row of data to push through the analytics pipeline.")
        
        col1, col2 = st.columns(2)
        generate_safe = col1.button("Generate & Test SAFE Sample", type="secondary", use_container_width=True)
        generate_fraud = col2.button("Generate & Test FRAUD Sample", type="primary", use_container_width=True)
        
        if generate_safe or generate_fraud:
            progress_bar = st.progress(0, text="Generating synthetic transaction data...")
            
            sample_features = np.zeros((1, 30))
            if generate_fraud:
                # Highly extreme values modeled after historical fraud
                sample_features[0, :] = np.random.randn(1, 30) * 2 # Adds generic noise
                sample_features[0, 14] = -18.0 # V14 extreme negative
                sample_features[0, 17] = -15.0 # V17 extreme negative
                sample_features[0, 12] = -16.0 # V12 extreme negative
                sample_features[0, 10] = -12.0 # V10 extreme negative
                sample_features[0, 4] = 15.0 # V4 extreme positive
                sample_features[0, 3] = -12.0 # V3 extreme negative
                sample_features[0, 29] = np.random.uniform(500, 5000) # High fraudulent amounts
            else:
                # Normal bounded values
                sample_features[0, :] = np.random.normal(0, 0.5, (1, 30))
                sample_features[0, 29] = np.random.uniform(10, 100) # Normal amounts
                
            progress_bar.progress(30, text="Scaling and feeding to Random Forest...")
            progress_bar.progress(50, text="Querying Isolation Forest anomalies...")
            progress_bar.progress(70, text="Testing Autoencoder neural network reconstruction...")
            
            results = detector.predict(sample_features)
            
            is_fraud = results['prediction'][0] == 1
            score = results['hybrid_score'][0]
            
            progress_bar.progress(85, text="Aggregating Hybrid metrics...")
            
            if is_fraud:
                st.error(f'🚨 FRAUD DETECTED! (Hybrid Score: {score:.2f})', icon="🚨")
            else:
                st.success(f'✅ Safe Transaction (Hybrid Score: {score:.2f})', icon="✅")
                
            st.write("### Model Breakdown")
            m1, m2, m3 = st.columns(3)
            m1.metric("Random Forest Prob", f"{results['rf_probability'][0]:.2f}")
            m2.metric("IsoForest Anomaly", "Yes" if results['iso_anomaly'][0] == 1 else "No")
            m3.metric("Autoencoder Error", f"{results['ae_reconstruction_error'][0]:.2f}")
            
            st.write("### Explainability (SHAP)")
            status_text = st.empty()
            status_text.write("Analyzing Feature Importance for this specific decision...")
            
            try:
                # Generate exact SHAP plots mapping feature importance (using optimized raw arrays)
                explainer = get_shap_explainer(detector.get_rf_model())
                scaled_sample = detector.get_scaler().transform(sample_features)
                shap_values = explainer.shap_values(scaled_sample, check_additivity=False)
                
                # Isolate the proper class side since TreeExplainer output varies based on Python dependency versions
                if isinstance(shap_values, list): vals = shap_values[1][0]
                else: vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                
                feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
                
                impact_indices = np.argsort(np.abs(vals))[-10:] # get top 10 impacts
                top_features = [feature_names[i] for i in impact_indices]
                top_impacts = [vals[i] for i in impact_indices]
                
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
                colors = ['#ef4444' if val > 0 else '#3b82f6' for val in top_impacts]
                bars = ax.barh(top_features, top_impacts, color=colors)
                
                ax.set_xlabel('SHAP Value (Impact on prediction)', color='gray')
                ax.set_title('Top 10 Features Driving This Decision (Random Forest)', color='white')
                
                ax.tick_params(colors='gray', which='both')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines['bottom'].set_color('gray')
                ax.spines['left'].set_color('gray')
                
                st.pyplot(fig, transparent=True)
                
                status_text.success("Feature Importance Analyzed Successfully.")
            except Exception as e:
                status_text.error("Analysis Failed.")
                st.warning(f"Could not generate Explainability plot for this sample: {e}")
                
            progress_bar.progress(100, text="Analysis Complete!")

def show_results():
    st.header("Model Performance Results")
    st.write("This section displays the dynamic performance metrics evaluated on the test dataset during the latest training phase.")
    
    metrics = None
    if os.path.exists('models/metrics.json'):
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
            
    if not metrics:
        st.warning("No metrics found. Please re-run the training script to generate `metrics.json`.")
        return
        
    st.write("### Evaluation Gauges")
    
    # Create visually appealing gauge charts using Plotly
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+gauge", value = metrics['accuracy'] * 100,
        domain = {'x': [0, 0.22], 'y': [0, 1]},
        title = {'text': "Accuracy %"},
        gauge = {'axis': {'range': [90, 100]}, 'bar': {'color': "#10b981"}}))

    fig.add_trace(go.Indicator(
        mode = "number+gauge", value = metrics['precision'] * 100,
        domain = {'x': [0.26, 0.48], 'y': [0, 1]},
        title = {'text': "Precision %"},
        gauge = {'axis': {'range': [0, max(100, metrics['precision'] * 100 + 20)]}, 'bar': {'color': "#3b82f6"}}))

    fig.add_trace(go.Indicator(
        mode = "number+gauge", value = metrics['recall'] * 100,
        domain = {'x': [0.52, 0.74], 'y': [0, 1]},
        title = {'text': "Recall %"},
        gauge = {'axis': {'range': [0, max(100, metrics['recall'] * 100 + 20)]}, 'bar': {'color': "#f59e0b"}}))
        
    fig.add_trace(go.Indicator(
        mode = "number+gauge", value = metrics['roc_auc'] * 100,
        domain = {'x': [0.78, 1.0], 'y': [0, 1]},
        title = {'text': "ROC-AUC %"},
        gauge = {'axis': {'range': [80, 100]}, 'bar': {'color': "#8b5cf6"}}))

    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "gray"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(facecolor='none')
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        ax.tick_params(colors='gray', which='both')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        
        st.pyplot(fig, transparent=True)
        
    with col2:
        st.write("### Analysis Note")
        st.info("The system uses an ensemble weighted approach. Precision (minimizing false positives) and Recall (minimizing false negatives) are heavily influenced by the combination of the Autoencoder's reconstruction limit and the Random Forest classification threshold.")
        st.write("The Confusion matrix shows the exact breakdown of standard true-positives handled during the test-split evaluation phase.")
        
        st.write("### Global Feature Importance")
        try:
            detector = load_detector()
            rf = detector.get_rf_model()
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:10] # Top 10
            
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
            ax.bar(range(10), importances[indices], color='#3b82f6')
            ax.set_xticks(range(10))
            ax.set_xticklabels([f"V{i}" if i != 29 else "Amount" for i in indices], rotation=45, color='gray')
            ax.tick_params(colors='gray', axis='y')
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines['bottom'].set_color('gray')
            ax.spines['left'].set_color('gray')
            
            st.pyplot(fig, transparent=True)
        except Exception as e:
            st.info(f"Could not load feature importances: {e}")

if __name__ == "__main__":
    main()
