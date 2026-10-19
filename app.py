import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load dataset and saved model
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

df = load_data()
model, scaler = load_model()

# Dashboard title
st.title("💳 Credit Card Fraud Detection System")
st.write("Analyze transactions and identify potentially fraudulent activity.")

st.divider()

# Dataset summary
total = len(df)
fraud_count = int((df["Class"] == 1).sum())
normal_count = int((df["Class"] == 0).sum())

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{total:,}")
col2.metric("Normal Transactions", f"{normal_count:,}")
col3.metric("Fraud Transactions", f"{fraud_count:,}")

st.divider()

# Transaction distribution chart
st.subheader("Transaction Distribution")

counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
counts.index = ["Normal", "Fraud"]

fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Transaction Type")
ax.set_ylabel("Number of Transactions")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)
st.subheader("📊 Transaction Analysis")

fraud_percentage = (fraud_count / total) * 100
normal_percentage = (normal_count / total) * 100

col1, col2 = st.columns(2)

col1.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")
col2.metric("Normal Percentage", f"{normal_percentage:.3f}%")

st.divider()
st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")
    st.write("Dataset Preview:")
    st.dataframe(uploaded_df.head())

    st.write("Total Rows:", len(uploaded_df))
    st.write("Total Columns:", len(uploaded_df.columns))
    if st.button("Predict Fraud for Uploaded Data"):
         if "Class" in uploaded_df.columns:
            features = uploaded_df.drop(columns=["Class"])
         else:
            features = uploaded_df.copy()

         if list(features.columns) != list(df.drop(columns=["Class"]).columns):
            st.error("CSV must contain the same transaction feature columns as the training dataset.")
         else:
            scaled_data = scaler.transform(features)
            predictions = model.predict(scaled_data)

            results = uploaded_df.copy()
            results["Predicted_Result"] = [
                "Fraud" if p == 1 else "Normal"
                for p in predictions
            ]

            st.subheader("Prediction Results")
            st.dataframe(results)
            st.write(
                "Potential Fraud Transactions:",
                int((predictions == 1).sum())
            )

# Transaction prediction
st.subheader("🔍 Test a Transaction")


st.write(
    "Enter transaction feature values to check the model's prediction. "
    "For a meaningful test, use values from the same dataset."
)

transaction_index = st.number_input(
    "Select transaction row number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("Check Transaction"):
    transaction = df.iloc[[int(transaction_index)]]

    actual_class = int(transaction["Class"].iloc[0])
    features = transaction.drop(columns=["Class"])

    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])

    probability = model.predict_proba(scaled_features)[0][1]

    if prediction == 1:
        st.error("Model Prediction: Potential Fraud")
    else:
        st.success("Model Prediction: Normal Transaction")

    st.write(f"Model fraud score: {probability:.2%}")

    st.write(
        "Dataset label:",
        "Fraud" if actual_class == 1 else "Normal"
    )

    st.caption(
        "This is an educational model, not a production fraud detection system."
    )