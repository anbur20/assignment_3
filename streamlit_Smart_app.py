import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run the training script first.")
    st.stop() # Stop the app if files are missing

st.title("💰 SmartPremium: Insurance Cost Predictor")
st.write("Enter the details below to get an estimated insurance premium.")

# Create input fields for numerical features from your dataset
age = st.slider("Age", 18, 100, 30)
annual_income = st.slider("Annual Income ($)", 0, 200000, 50000)
num_dependents = st.slider("Number of Dependents", 0, 10, 0)
health_score = st.slider("Health Score", 0, 100, 50)
previous_claims = st.slider("Previous Claims", 0, 10, 0)
vehicle_age = st.slider("Vehicle Age (Years)", 0, 25, 5)
credit_score = st.slider("Credit Score", 300, 850, 700)
insurance_duration = st.slider("Insurance Duration (Years)", 0, 50, 5)

# Create input fields for categorical features from your dataset
gender = st.selectbox("Gender", ['Male', 'Female'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
education_level = st.selectbox("Education Level", ['High School', 'College', 'Bachelor', 'Master', 'PhD'])
occupation = st.selectbox("Occupation", ['Student', 'Employed', 'Unemployed', 'Retired'])
location = st.selectbox("Location", ['Urban', 'Rural', 'Suburban'])
policy_type = st.selectbox("Policy Type", ['Standard', 'Silver', 'Gold', 'Platinum'])
customer_feedback = st.selectbox("Customer Feedback", ['Good', 'Average', 'Poor'])
smoking_status = st.selectbox("Smoking Status", ['Yes', 'No'])
property_type = st.selectbox("Property Type", ['House', 'Apartment', 'Condo'])
exercise_frequency = st.selectbox("Exercise Frequency", ['Rarely', 'Occasionally', 'Regularly'])

# Prepare user input for prediction
# This must match the column names and order from your training data after one-hot encoding
input_data = {
    'Age': age,
    'Annual Income': annual_income,
    'Number of Dependents': num_dependents,
    'Health Score': health_score,
    'Previous Claims': previous_claims,
    'Vehicle Age': vehicle_age,
    'Credit Score': credit_score,
    'Insurance Duration': insurance_duration,
    # One-hot encoded categorical features
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Marital Status_Married': 1 if marital_status == 'Married' else 0,
    'Marital Status_Single': 1 if marital_status == 'Single' else 0,
    'Education Level_Bachelor': 1 if education_level == 'Bachelor' else 0,
    'Education Level_College': 1 if education_level == 'College' else 0,
    'Education Level_High School': 1 if education_level == 'High School' else 0,
    'Education Level_Master': 1 if education_level == 'Master' else 0,
    'Education Level_PhD': 1 if education_level == 'PhD' else 0,
    'Occupation_Employed': 1 if occupation == 'Employed' else 0,
    'Occupation_Retired': 1 if occupation == 'Retired' else 0,
    'Occupation_Student': 1 if occupation == 'Student' else 0,
    'Occupation_Unemployed': 1 if occupation == 'Unemployed' else 0,
    'Location_Rural': 1 if location == 'Rural' else 0,
    'Location_Suburban': 1 if location == 'Suburban' else 0,
    'Location_Urban': 1 if location == 'Urban' else 0,
    'Policy Type_Gold': 1 if policy_type == 'Gold' else 0,
    'Policy Type_Platinum': 1 if policy_type == 'Platinum' else 0,
    'Policy Type_Silver': 1 if policy_type == 'Silver' else 0,
    'Policy Type_Standard': 1 if policy_type == 'Standard' else 0,
    'Customer Feedback_Good': 1 if customer_feedback == 'Good' else 0,
    'Customer Feedback_Poor': 1 if customer_feedback == 'Poor' else 0,
    'Smoking Status_Yes': 1 if smoking_status == 'Yes' else 0,
    'Property Type_Condo': 1 if property_type == 'Condo' else 0,
    'Property Type_House': 1 if property_type == 'House' else 0,
    'Exercise Frequency_Occasionally': 1 if exercise_frequency == 'Occasionally' else 0,
    'Exercise Frequency_Rarely': 1 if exercise_frequency == 'Rarely' else 0
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

if st.button("Predict Premium"):
    # Ensure all columns exist and are in the correct order for the scaler
    # This list must match the columns in your training set *exactly*
    #expected_columns = list(scaler.mean_.index)
    expected_columns = list(scaler.mean_)

    # Reindex the input DataFrame to match the expected column order
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_scaled)

    st.markdown(f"### **Estimated Premium: ${prediction[0]:,.2f}** 💲")