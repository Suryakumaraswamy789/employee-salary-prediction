import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
#model = joblib.load("best_model.pkl")
import cloudpickle

# Load trained model
with open('salary_model.pkl', 'rb') as f:
    model = cloudpickle.load(f)


# Streamlit config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="🧑‍💼", layout="centered")
st.title("Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.number_input("Age", 18, 100, 30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                                'Local-gov', 'State-gov', 'Notlisted'])  # '?' replaced by 'notlisted'
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 50000)
education_num = st.sidebar.number_input("Education Num", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                          'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                                  'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                                  'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others'])
relationship = st.sidebar.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.sidebar.number_input("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany',
                                                         'Canada', 'England', 'China', 'Cuba', 'Iran', 'Other'])

# Format input as DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Show input
st.subheader("Input Data Preview")
st.write(input_data)

# Label Map
label_map = {'<=50K': "≤50K", '>50K': ">50K"}

# Predict
if st.button("Predict Salary Class"):
    try:
        st.write("Predicting...")
        prediction = model.predict(input_data)[0]
        st.success(f"Prediction: {label_map.get(prediction, prediction)}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Batch prediction
st.markdown("### Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with same column names", type=['csv'])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(batch_df)

        batch_preds = model.predict(batch_df)
        batch_preds_df = pd.DataFrame({'Predicted Salary Class': [label_map.get(p, p) for p in batch_preds]})
        st.write("Predictions:")
        st.write(batch_preds_df)

        csv = batch_preds_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error in batch prediction: {str(e)}")
