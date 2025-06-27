import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model=joblib.load('predict_salary.pkl')
scaler=joblib.load('scaler.pkl')
#design the layout of our Basic app
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Predictor App")
st.subheader("Predict the Salary based on Years of Experience")
st.write("select the years of experience to see the estimation salary")

#create a drop down for years of experience
years = [x for x in range(0, 20)] #list confrihancive
years_exp=st.selectbox("Years of Experience", years)

if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    input_scaled=scaler.transform(input_data)
    predicted_salary = model.predict(input_scaled)
    st.success(f"Predicted Salary: Rs. {predicted_salary[0]:,.2f}")