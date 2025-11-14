import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Transaction Guard AI", layout="centered")

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #0062E6, #33AEFF);
            color: white;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        .stFileUploader {
            background: rgba(240, 248, 255, 0.7);
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        h3 { color: #1a73e8; }
        .footer {
            text-align: center;
            padding: 15px;
            margin-top: 40px;
            border-top: 1px solid #eaeaea;
            color: #555;
            font-size: 15px;
        }
        .divider {
            border-top: 2px solid #e3e3e3;
            margin: 30px 0;
        }
        .card {
            padding: 18px;
            border-radius: 14px;
            color: #000;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>💳 Transaction Guard AI</h1>
    <h3>An AI-powered Transaction Fraud Detection System</h3>
    <p style='margin-top:10px; font-size:15px;'>Developed by <b>Mantri Diwakar</b></p>
</div>
""", unsafe_allow_html=True)

# ==================== STEP 1: UPLOAD TRAINING DATA ====================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("### 🧠 Step 1: Upload Training Dataset (must include a 'Fraud' column)")
train_file = st.file_uploader("Upload Training Dataset", type=["csv"], key="train")

model = None
label_encoders = {}

if train_file is not None:
    train_data = pd.read_csv(train_file)
    st.success("✅ Training dataset uploaded successfully!")
    st.dataframe(train_data.head())

    if 'Fraud' not in train_data.columns:
        st.error("The dataset must contain a column named 'Fraud'. Please upload the correct file.")
    else:
        # Encode categorical features
        df = train_data.copy()
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop('Fraud', axis=1)
        y = df['Fraud']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Progress animation
        st.markdown("#### 🚀 Training the Model...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        # Train model
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 2)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred) * 100

        st.success(f"🎉 Model training completed successfully in {training_time} seconds!")
        st.info(f"Validation Accuracy: **{acc:.2f}%**")

# ==================== STEP 2: UPLOAD TEST DATA ====================
if model is not None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Step 2: Upload Test Dataset (without 'Fraud' column)")
    test_file = st.file_uploader("Upload Test Dataset", type=["csv"], key="test")

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.success("✅ Test dataset uploaded successfully!")
        st.dataframe(test_data.head())

        test_df = test_data.copy()

        # Handle categorical encodings safely
        for col in test_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                le = label_encoders[col]

                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')

                test_df[col] = le.transform(test_df[col])
            else:
                test_df[col] = 'unknown'
                le = LabelEncoder()
                le.fit(['unknown'])
                test_df[col] = le.transform(test_df[col])

        # Predictions
        preds = model.predict(test_df)
        probs = model.predict_proba(test_df)[:, 1]

        results = test_data.copy()
        results['Predicted_Fraud'] = preds
        results['Fraud_Probability'] = probs

        # ==================== SUMMARY CARDS ====================
        total_tx = len(results)
        fraud_tx = results['Predicted_Fraud'].sum()
        non_fraud_tx = total_tx - fraud_tx
        fraud_percent = round((fraud_tx / total_tx) * 100, 2)

        st.markdown("### 📌 Summary of Predictions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="card" style="background:#e8f4ff; border-left:6px solid #1a73e8;">
                📦<br>Total<br><span style="font-size:22px;">{total_tx}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card" style="background:#ffecec; border-left:6px solid #ff4c4c;">
                ⚠️<br>Fraud<br><span style="font-size:22px;">{fraud_tx}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card" style="background:#e9ffec; border-left:6px solid #22c55e;">
                ✔️<br>Non-Fraud<br><span style="font-size:22px;">{non_fraud_tx}</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="card" style="background:#fff8e6; border-left:6px solid #f5a623;">
                📊<br>Fraud %<br><span style="font-size:22px;">{fraud_percent}%</span>
            </div>
            """, unsafe_allow_html=True)

        # ==================== PREDICTION TABLE ====================
        st.markdown("### 🔍 Prediction Results")
        st.dataframe(results.head(15))

        csv_download = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv_download, "fraud_predictions.csv", "text/csv")

# ==================== FOOTER ====================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>💡 Developed by <b>Mantri Diwakar</b> | Powered by AI & Streamlit</div>", unsafe_allow_html=True)

