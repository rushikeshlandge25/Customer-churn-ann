import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ---------------- Load Files ----------------
model = load_model("model.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900, 600)

geography = st.selectbox(
    "Geography",
    onehot_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder.classes_
)

age = st.slider("Age", 18, 92, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", value=0.0)

num_products = st.slider("Number of Products", 1, 4, 1)

has_card = st.selectbox("Has Credit Card", [0, 1])

active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", value=50000.0)

# ---------------- Prediction ----------------

if st.button("Predict"):

    gender = label_encoder.transform([gender])[0]

    geo = onehot_encoder.transform([[geography]])

    if hasattr(geo, "toarray"):
        geo = geo.toarray()

    geo_df = pd.DataFrame(
        geo,
        columns=onehot_encoder.get_feature_names_out(["Geography"])
    )

    input_df = pd.DataFrame({
        "CreditScore":[credit_score],
        "Gender":[gender],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[num_products],
        "HasCrCard":[has_card],
        "IsActiveMember":[active],
        "EstimatedSalary":[salary]
    })

    final_df = pd.concat([input_df, geo_df], axis=1)

    # Scale
    final_df = scaler.transform(final_df)

    prediction = model.predict(final_df)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(f"Churn Probability : {probability:.2%}")

    if probability > 0.5:
        st.error("⚠ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")