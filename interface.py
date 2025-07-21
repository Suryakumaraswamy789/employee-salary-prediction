import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
workclass_encoder = joblib.load("workclass_encoder.pkl")
marital_encoder = joblib.load("marital_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
relationship_encoder = joblib.load("relationship_encoder.pkl")
race_encoder = joblib.load("race_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
country_encoder = joblib.load("country_encoder.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction App")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)

workclass = st.sidebar.selectbox("Workclass", list(workclass_encoder.classes_))
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
marital_status = st.sidebar.selectbox("Marital Status", list(marital_encoder.classes_))
occupation = st.sidebar.selectbox("Occupation", list(occupation_encoder.classes_))
relationship = st.sidebar.selectbox("Relationship", list(relationship_encoder.classes_))
race = st.sidebar.selectbox("Race", list(race_encoder.classes_))
gender = st.sidebar.selectbox("Gender", list(gender_encoder.classes_))
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", list(country_encoder.classes_))

# Encode categorical inputs
education_mapping = {
    "Bachelors": 13,
    "Masters": 14,
    "PhD": 16,
    "HS-grad": 9,
    "Assoc": 12,
    "Some-college": 10
}
educational_num = education_mapping[education]

# Create input dataframe with all 13 required features
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_encoder.transform([workclass])[0]],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_encoder.transform([marital_status])[0]],
    'occupation': [occupation_encoder.transform([occupation])[0]],
    'relationship': [relationship_encoder.transform([relationship])[0]],
    'race': [race_encoder.transform([race])[0]],
    'gender': [gender_encoder.transform([gender])[0]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [country_encoder.transform([native_country])[0]],
})

# Display the input data
st.write("### Input Data")
st.dataframe(input_data)

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_data)
    st.success(f"🧠 Prediction: {prediction[0]}")
