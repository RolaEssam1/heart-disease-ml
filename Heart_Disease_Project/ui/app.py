import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Risk Prediction")
st.write("Enter patient details and get a prediction from the trained model.")

# loading model
@st.cache_resource
def load_model():
    model = joblib.load("../models/final_model.pkl")
    return model

model = load_model()

age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type", [1, 2, 3, 4])
trestbps = st.number_input("Resting blood pressure (mm Hg)", value=130)
chol = st.number_input("Serum cholesterol (mg/dl)", value=250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum heart rate achieved", value=150)
exang = st.selectbox("Exercise induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, format="%.1f")
slope = st.selectbox("Slope of ST segment", [1, 2, 3])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [3, 6, 7], format_func=lambda x: {3:"Normal", 6:"Fixed defect", 7:"Reversible defect"}[x])

input_df = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}])

st.subheader("Input Summary")
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Model predicts **Heart Disease** risk with probability {proba:.2f}")
    else:
        st.success(f" Model predicts **No Heart Disease** with probability {1-proba:.2f}")
