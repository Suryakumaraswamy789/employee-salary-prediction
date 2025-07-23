import numpy as np
import streamlit as st
import pandas as pd
import joblib
import cloudpickle
with open('salary_model.pkl','rb') as f:
    model =cloudpickle.load(f)

st.set_page_config(page_title="Employee Salary Prediction",page_icon="",layout="centered")
st.title("employee Salary Prediction App")
st.markdown("predict wheatheran employee earn >50k or <50k based on input feature")
st.sidebar.header("Input Employee Details")
age = st.sidebar.number_input("Age", 18, 100, 30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                       'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 50000)
education = st.sidebar.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 
                                       'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th', '10th', '1st-4th', 'Preschool'])
education_num = st.sidebar.number_input("Educational-num", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 
                                                 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
                                         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
                                         'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.sidebar.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.sidebar.number_input("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 
                                                 'Canada', 'England', 'China', 'Cuba', 'Iran', 'Other'])

# Encode categorical variables simply (use same mapping as used during training)
def encode_input():
    workclass_map = {k: i for i, k in enumerate(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])}
    education_map = {k: i for i, k in enumerate(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 
                                                 'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th', '10th', '1st-4th', 'Preschool'])}
    marital_map = {k: i for i, k in enumerate(['Married-civ-spouse', 'Divorced', 'Never-married', 
                                               'Separated', 'Widowed', 'Married-spouse-absent'])}
    occupation_map = {k: i for i, k in enumerate(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
                                                  'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
                                                  'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])}
    relationship_map = {k: i for i, k in enumerate(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])}
    race_map = {k: i for i, k in enumerate(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])}
    gender_map = {'Male': 1, 'Female': 0}
    country_map = {k: i for i, k in enumerate(['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 
                                               'Canada', 'England', 'China', 'Cuba', 'Iran', 'Other'])}

    return np.array([
        age,
        workclass_map[workclass],
        fnlwgt,
        education_map[education],
        education_num,
        marital_map[marital_status],
        occupation_map[occupation],
        relationship_map[relationship],
        race_map[race],
        gender_map[gender],
        capital_gain,
        capital_loss,
        hours_per_week,
        country_map[native_country]
    ]).reshape(1, -1)


input_df = encode_input()
st.write("## Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f'Prediction: {prediction[0]}')

# Batch prediction
st.markdown("### Batch Prediction")
st.markdown("Upload a CSV file for batch prediction:")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.write(batch_data)
    
    batch_preds = model.predict(batch_data)
    batch_preds = [str(pred) for pred in batch_preds]
    st.write("Predictions:")
    st.write(batch_preds)
    # Batch predictions as downloadable CSV
    batch_preds_df = pd.DataFrame(batch_preds, columns=['Predicted Salary Class'])
    st.write(batch_preds_df)

    csv = batch_preds_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Predictions CSV', csv, file_name='predicted_classes.csv', mime='text/csv')
