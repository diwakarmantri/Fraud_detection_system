import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Transaction Guard AI", layout="wide")

# ----------------------------
# Custom CSS for Styling
# ----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
            color: #1a202c;
        }
        h1, h4 {
            text-align: center;
        }
        .metric-card {
            padding: 20px;
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 30px;
        }
        hr {
            border: 1px solid #e2e8f0;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1 style='color:#2b6cb0;'>Transaction Guard AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:gray;'>An AI-powered Transaction Fraud Detection System</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📂 Upload Datasets")

train_file = st.file_uploader("Upload Training Dataset (must include 'Fraud' column)", type=["csv"])
test_file = st.file_uploader("Upload Test Dataset (must include 'Fraud' column)", type=["csv"])

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Check 'Fraud' column
    if 'Fraud' not in train_df.columns:
        st.error("❌ The training dataset must contain a column named 'Fraud'. Please rename and re-upload.")
    elif 'Fraud' not in test_df.columns:
        st.error("❌ The test dataset must contain a column named 'Fraud'. Please rename and re-upload.")
    else:
        st.success("✅ Datasets uploaded successfully!")

        # ----------------------------
        # Data Preview
        # ----------------------------
        st.subheader("📊 Dataset Preview")
        st.write("**Training Dataset:**", train_df.head())
        st.write("**Test Dataset:**", test_df.head())

        # ----------------------------
        # Split Data
        # ----------------------------
        X_train = train_df.drop(columns=['Fraud'])
        y_train = train_df['Fraud']

        X_test = test_df.drop(columns=['Fraud'])
        y_test = test_df['Fraud']

        # ----------------------------
        # Feature Scaling
        # ----------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ----------------------------
        # Model Training
        # ----------------------------
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # ----------------------------
        # Predictions
        # ----------------------------
        y_pred = model.predict(X_test_scaled)

        # ----------------------------
        # Performance Metrics
        # ----------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ----------------------------
        # Display Metrics as Cards
        # ----------------------------
        st.subheader("📈 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2>{accuracy:.3f}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Precision</h3><h2>{precision:.3f}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Recall</h3><h2>{recall:.3f}</h2></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3>F1 Score</h3><h2>{f1:.3f}</h2></div>", unsafe_allow_html=True)

        # ----------------------------
        # Classification Report
        # ----------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🧾 Detailed Classification Report")
        st.text(classification_report(y_test, y_pred))

        # ----------------------------
        # Footer
        # ----------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='footer'>Developed by <b>Mantri Diwakar</b></p>", unsafe_allow_html=True)

else:
    st.info("👆 Please upload both training and test CSV files to proceed.")

