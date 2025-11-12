import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Transaction Guard AI",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# PAGE HEADER
# ----------------------------
st.markdown("""
    <div style="background:linear-gradient(90deg, #e3f2fd, #bbdefb);
                padding:25px; border-radius:12px; text-align:center;">
        <h1 style="color:#0d47a1; margin-bottom:5px;">💳 Transaction Guard AI</h1>
        <h3 style="color:#1565C0; margin-top:0;">An AI-powered Transaction Fraud Detection System</h3>
        <p style="text-align:right; color:#1a237e; font-weight:bold; font-size:16px;">Developed by Mantri Diwakar</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# STEP 1: TRAINING DATA UPLOAD
# ----------------------------
st.markdown("### 🧠 Step 1: Upload Training Dataset (with `fraud` column)")
train_file = st.file_uploader("Upload training file (CSV/XLSX)", type=["csv", "xlsx"], key="train")

if train_file:
    if train_file.name.endswith('.csv'):
        train_df = pd.read_csv(train_file)
    else:
        train_df = pd.read_excel(train_file)

    if 'fraud' not in train_df.columns:
        st.error("❌ Training dataset must contain a column named 'fraud'")
        st.stop()

    st.success("✅ Training dataset uploaded successfully!")
    st.dataframe(train_df.head())

    # Training section
    st.markdown("### 🚀 Training the Model...")
    start_time = time.time()

    X = train_df.drop(columns=['fraud'])
    y = train_df['fraud']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    train_duration = time.time() - start_time
    st.success(f"🎉 Model training completed successfully in {train_duration:.2f} seconds!")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    st.write("#### Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    st.write("#### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # ----------------------------
    # STEP 2: TEST DATA UPLOAD
    # ----------------------------
    st.markdown("---")
    st.markdown("### 📂 Step 2: Upload Test Dataset (without `fraud` column)")
    test_file = st.file_uploader("Upload test file (CSV/XLSX)", type=["csv", "xlsx"], key="test")

    if test_file:
        if test_file.name.endswith('.csv'):
            test_df = pd.read_csv(test_file)
        else:
            test_df = pd.read_excel(test_file)

        st.success("✅ Test dataset uploaded successfully!")
        st.dataframe(test_df.head())

        # Align columns with trained model
        test_encoded = pd.get_dummies(test_df, drop_first=True)
        missing_cols = set(X.columns) - set(test_encoded.columns)
        for c in missing_cols:
            test_encoded[c] = 0
        test_encoded = test_encoded[X.columns]

        # Predict frauds
        predictions = model.predict(test_encoded)
        test_df['Predicted_Fraud'] = predictions

        fraud_cases = test_df[test_df['Predicted_Fraud'] == 1]
        total_transactions = len(test_df)
        fraud_count = len(fraud_cases)

        st.markdown("### 📊 Detection Summary")
        col1, col2 = st.columns(2)
        col1.metric("Total Transactions", total_transactions)
        col2.metric("Fraud Detected", fraud_count)

        # Fraud table and download option
        st.markdown("### ⚠️ Fraudulent Transactions Detected")
        st.dataframe(fraud_cases.head(10))

        csv = fraud_cases.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Fraudulent Transactions",
            data=csv,
            file_name="fraud_detected.csv",
            mime="text/csv"
        )

        # ----------------------------
        # STEP 3: DATA VISUALIZATION
        # ----------------------------
        st.markdown("---")
        st.subheader("📈 Data Visualization Dashboard")

        numeric_cols = test_df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 0:
            st.markdown("#### 📊 Create Custom Graphs")
            graph_type = st.selectbox("Select Graph Type", ["Histogram", "Scatter", "Boxplot"])
            x_axis = st.selectbox("Select X-axis", numeric_cols)
            y_axis = st.selectbox("Select Y-axis (optional)", [None] + numeric_cols)

            plt.figure(figsize=(7, 4))
            if graph_type == "Histogram":
                sns.histplot(test_df[x_axis], bins=30, kde=True, color="#42a5f5")
            elif graph_type == "Scatter" and y_axis:
                sns.scatterplot(x=test_df[x_axis], y=test_df[y_axis], color="#1e88e5")
            elif graph_type == "Boxplot" and y_axis:
                sns.boxplot(x=test_df[x_axis], y=test_df[y_axis], palette="Blues")
            else:
                st.warning("Please select a valid combination for the selected graph.")
            st.pyplot(plt)

