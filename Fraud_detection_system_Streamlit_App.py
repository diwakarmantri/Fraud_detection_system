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
        # Encode all categorical features
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
        end_time = time.time()
        training_time = round(end_time - start_time, 2)

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

        # Robustly handle all categorical columns
        for col in test_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                le = label_encoders[col]

                # Replace unseen labels with 'unknown'
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

                # Add 'unknown' to classes_ if not present (keep as np.array)
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')

                test_df[col] = le.transform(test_df[col])
            else:
                # New column in test data: encode all values as 'unknown'
                test_df[col] = 'unknown'
                le = LabelEncoder()
                le.fit(['unknown'])
                test_df[col] = le.transform(test_df[col])

        # Make predictions
        preds = model.predict(test_df)
        probs = model.predict_proba(test_df)[:, 1]

        results = test_data.copy()
        results['Predicted_Fraud'] = preds
        results['Fraud_Probability'] = probs

        st.markdown("### 🔍 Prediction Results")
        st.dataframe(results.head(15))

        csv_download = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv_download, "fraud_predictions.csv", "text/csv")

# ==================== FOOTER ====================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>💡 Developed by <b>Mantri Diwakar</b> | Powered by AI & Streamlit</div>", unsafe_allow_html=True)

