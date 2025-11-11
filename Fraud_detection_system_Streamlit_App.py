import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Transaction Guard AI — An AI-powered Transaction Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# PAGE HEADER
# ----------------------------
st.markdown("""
    <div style="
        background:linear-gradient(90deg, #e3f2fd, #bbdefb);
        padding:20px;
        border-radius:12px;
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
    ">
        <div style="flex:1;">
            <h1 style="color:#0d47a1; margin-bottom:5px;">💳 Transaction Guard AI</h1>
            <h3 style="color:#1565C0; margin-top:0;">An AI-powered Transaction Fraud Detection System</h3>
        </div>
        <div style="text-align:right; color:#1a237e; font-weight:bold; font-size:16px; white-space:nowrap;">
            Developed by <br>– Mantri Diwakar
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("")

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload Transaction Data", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # ----------------------------
    # DATA VISUALIZATION SECTION
    # ----------------------------
    st.markdown("---")
    st.subheader("📊 Data Visualization Dashboard")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💰 Transaction Amount Distribution")
            plt.figure(figsize=(6, 4))
            sns.histplot(df[numeric_cols[0]], bins=30, kde=True, color="#42a5f5")
            st.pyplot(plt)
            plt.clf()
        with col2:
            st.markdown("#### 📈 Correlation Heatmap")
            plt.figure(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues")
            st.pyplot(plt)
            plt.clf()

    # ----------------------------
    # FRAUD DETECTION MODEL SECTION
    # ----------------------------
    st.markdown("---")
    st.subheader("🤖 AI-Powered Fraud Detection")

    if 'Is_Fraud' in df.columns:
        st.info("🧠 Detected label column: `Is_Fraud`. Training AI model...")

        X = df.drop(columns=['Is_Fraud'])
        y = df['Is_Fraud']

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.markdown("### 📊 Model Performance Summary")
        st.metric("✅ Accuracy", f"{accuracy*100:.2f}%")

        st.markdown("#### 🔹 Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

        st.markdown("#### 📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.warning("⚠️ Column `Is_Fraud` not found. Running basic rule-based detection instead.")

        threshold = st.slider("Set Amount Threshold for Flagging Fraud 💰", 1000, 100000, 50000)
        df['Predicted_Fraud'] = np.where(df['Transaction_Amount'] > threshold, 1, 0)

        st.markdown("### ⚙️ Rule-Based Detection Results")
        st.dataframe(df[['Transaction_ID', 'Transaction_Amount', 'Predicted_Fraud']].head(10))

else:
    # ----------------------------
    # DEFAULT HOME PAGE (NO FILE)
    # ----------------------------
    st.markdown("""
    <div style="background-color:#e3f2fd; padding:25px; border-radius:15px; text-align:center;">
        <h3>🚀 Required Dataset Columns</h3>
        <p style="font-size:17px; color:#0d47a1;">Ensure your dataset includes the following columns before uploading:</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; justify-content:center;">
    <table style="border-collapse: collapse; text-align:center; font-size:16px;">
        <tr style="background-color:#bbdefb; color:#0d47a1;">
            <th style="padding:10px; border:1px solid #90caf9;">🧾 Column</th>
            <th style="padding:10px; border:1px solid #90caf9;">📘 Description</th>
        </tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">🆔 <b>Transaction_ID</b></td><td style="padding:10px; border:1px solid #bbdefb;">Unique transaction identifier</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">👤 <b>Account_Number</b></td><td style="padding:10px; border:1px solid #bbdefb;">Account or customer ID</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">📅 <b>Transaction_Date</b></td><td style="padding:10px; border:1px solid #bbdefb;">Date and time of transaction</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">💰 <b>Transaction_Amount</b></td><td style="padding:10px; border:1px solid #bbdefb;">Transaction amount</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">🏪 <b>Merchant_ID</b></td><td style="padding:10px; border:1px solid #bbdefb;">Merchant identifier</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">💻 <b>Device_ID</b></td><td style="padding:10px; border:1px solid #bbdefb;">Device used for transaction</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">🔄 <b>Transaction_Type</b></td><td style="padding:10px; border:1px solid #bbdefb;">Mode (Online / Offline)</td></tr>
        <tr><td style="padding:10px; border:1px solid #bbdefb;">⚠️ <b>Is_Fraud</b> *(optional)*</td><td style="padding:10px; border:1px solid #bbdefb;">1 = Fraud, 0 = Legitimate</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# FOOTER LINE
# ----------------------------
st.markdown("---")

