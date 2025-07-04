import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to predict diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                              columns=X.columns)
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.error(f"🔴 The person is likely diabetic. (Confidence: {probability:.2f})")
    else:
        st.success(f"🟢 The person is likely not diabetic. (Confidence: {1 - probability:.2f})")
