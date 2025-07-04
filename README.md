# 🩺 Diabetes Prediction Web App

A machine learning-based Streamlit web application to predict whether a person is diabetic or not using health parameters like glucose level, insulin, BMI, age, and more.

## 🔍 Features
- Built using **Logistic Regression**
- Powered by **scikit-learn**, **pandas**, and **Streamlit**
- Uses the **PIMA Indian Diabetes Dataset**
- Deployed live on Streamlit Cloud ✅

## 🌐 Live App
🔗 [Click here to try the app](https://diabetes-prediction-app-ngecphawf5aknkdky36pr3.streamlit.app)

📘 Model Development Notebook:
→ diabetes_model_dev.ipynb (Google Colab)

## 📊 Input Parameters
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age

## 💡 Model Info
- Dataset Imbalance handled using `class_weight='balanced'`
- StandardScaler applied for normalization
- 80/20 train-test split

## 🚀 How to Run Locally
```bash
git clone https://github.com/yashshingne33/diabetes-prediction-app.git
cd diabetes-prediction-app
pip install -r requirements.txt
streamlit run app.py
