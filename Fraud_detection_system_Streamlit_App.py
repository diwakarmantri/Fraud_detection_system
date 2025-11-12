import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import io

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection System using Machine Learning")
st.write("Train the model using a dataset containing a **fraud column**, then test it on a new dataset without that column.")

st.sidebar.header("Steps")
st.sidebar.markdown("""
1️⃣ Upload training dataset (with `fraud` column)  
2️⃣ Train the model  
3️⃣ Upload testing dataset (without `fraud` column)  
4️⃣ Predict fraud transactions  
""")

# --- Step 1: Upload training dataset ---
st.subheader("Step 1: Upload Training Dataset (with 'fraud' column)")
train_file = st.file_uploader("Upload your training CSV file", type=["csv"], key="train")

model = None

if train_file is not None:
    df_train = pd.read_csv(train_file)
    st.success(f"✅ Training dataset uploaded with {df_train.shape[0]} rows and {df_train.shape[1]} columns.")
    st.dataframe(df_train.head())

    if "fraud" not in df_train.columns:
        st.error("❌ The dataset must contain a column named 'fraud'. Please re-upload the correct file.")
    else:
        # Split features and target
        X = df_train.drop("fraud", axis=1)
        y = df_train["fraud"]

        # Handle categorical columns if any
        X = pd.get_dummies(X, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("🚀 Train Model"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save model to memory
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            st.session_state['trained_model'] = buffer.getvalue()
            st.session_state['columns'] = X.columns.tolist()

            # Evaluate model
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.success("✅ Model training completed successfully!")
            st.write("### Model Performance")
            st.dataframe(pd.DataFrame(report).transpose())

            cm = confusion_matrix(y_test, y_pred)
            st.write("### Confusion Matrix")
            st.write(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

# --- Step 2: Upload testing dataset ---
st.subheader("Step 2: Upload Testing Dataset (without 'fraud' column)")
test_file = st.file_uploader("Upload your testing CSV file", type=["csv"], key="test")

if test_file is not None:
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model first by uploading a training dataset.")
    else:
        df_test = pd.read_csv(test_file)
        st.success(f"✅ Testing dataset uploaded with {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
        st.dataframe(df_test.head())

        # Load model and align columns
        model = pickle.loads(st.session_state['trained_model'])
        model_cols = st.session_state['columns']

        X_test = pd.get_dummies(df_test, drop_first=True)

        # Align with training columns
        for col in model_cols:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[model_cols]

        # Predict fraud probabilities
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        df_test["Predicted_Fraud"] = preds
        df_test["Fraud_Probability"] = probs

        st.write("### 🔍 Prediction Results")
        st.dataframe(df_test.head(20))

        fraud_cases = df_test[df_test["Predicted_Fraud"] == 1]
        st.write(f"### ⚠️ Detected Fraudulent Transactions: {fraud_cases.shape[0]}")
        st.dataframe(fraud_cases)

        # Download option
        csv = df_test.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

