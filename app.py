import streamlit as st
import joblib
import numpy as np

st.title("🎓 Student Performance Prediction")

# Load pre-trained model and scaler
model = joblib.load("model/student_model(1).pkl")
scaler = joblib.load("model/scaler(1).pkl")

# Input fields as select options
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.selectbox("Age", list(range(10, 26)))  # 10 to 25 years
study_hours = st.selectbox("Study Hours per Day", list(range(0, 11)))  # 0 to 10 hours
attendance = st.selectbox("Attendance (%)", list(range(0, 101, 5)))  # 0, 5, 10,... 100
previous_marks = st.selectbox("Previous Marks", list(range(0, 101, 5)))  # 0,5,10...100
internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
family_support = st.selectbox("Family Support", ["Yes", "No"])

# Convert categorical inputs to numerical
gender_num = 1 if gender == "Male" else 0
internet_num = 1 if internet == "Yes" else 0
family_support_num = 1 if family_support == "Yes" else 0

if st.button("Predict"):
    # Create data array
    data = np.array([[gender_num, age, study_hours, attendance, previous_marks, internet_num, family_support_num]])
    
    # Scale data
    data_scaled = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data_scaled)
    
    if prediction[0] == 1:
        st.success("✅ Student Will Pass")
    else:
        st.error("❌ Student May Fail")
