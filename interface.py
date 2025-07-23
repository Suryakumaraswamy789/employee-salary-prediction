import streamlit as st
import pandas as pd
import joblib
import cloudpickle
with open('salary_model.pkl','rb') as f:
    model =cloudpickle.load(f)
print(type(model))
print(dir(model))
st.set_page_config(page_title="Employee Salary Prediction",page_icon="",layout="centered")
st.title("employee Salary Prediction App")
st.markdown("predict wheatheran employee earn >50k or <50k based on input feature")
st.sidebar.header("Input Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education_num = st.sidebar.number_input("Education Number (Years of education)", min_value=1, max_value=16, value=10)
marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
hours_per_week = st.sidebar.number_input("Hours per week", min_value=1, max_value=100, value=40)
gender=st.sidebar.number_input("gender",min_value=1, max_value=0, value=1)


# Build input DataFrame (must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [education_num],
    'occupation': [occupation],
    'gender':[gender],
    'hours-per-week': [hours_per_week],
    'experience': [workclass]
})
def preprocess_input(age, workclass, education_num, marital_status, occupation, hours_per_week):
    # For demonstration, let's convert categorical to simple numeric encoding.
    # Replace this with your real preprocessing pipeline.

    workclass_map = {k:i for i,k in enumerate(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])}
    marital_map = {k:i for i,k in enumerate(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed'])}
    occupation_map = {k:i for i,k in enumerate(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])}

    processed = np.array([
        age,
        workclass_map[workclass],
        education_num,
        marital_map[marital_status],
        occupation_map[occupation],
        hours_per_week
    ]).reshape(1, -1)

    return processed
input_df=preprocess_input(age, workclass, education_num, marital_status, occupation, hours_per_week)
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
