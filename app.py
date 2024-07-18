import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

st.title("Heart Disease Prediction")

# Define the input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)

# Map sex to numerical values
sex = 1 if sex == "Male" else 0

# Make a prediction
if st.button("Predict"):
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    new_data = scalar.transform(input_data)
    prediction = regmodel.predict(new_data)
    st.write("Prediction: You have heart disease" if prediction[0] == 1 else "Prediction: You do not have heart disease")
