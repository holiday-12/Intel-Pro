import streamlit as st
import pickle
import numpy as np
import requests

def load_model():
    with open('All-Model/ML/save_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

st.title("Software Developer Salary Prediction")
st.write("""### We need some information to predict the salary""")

countries = (
    "United States of America",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "Ukraine",
    "India",
    "France",
    "Canada",
    "Brazil",
    "Spain",
    "Italy",
    "Netherlands",
    "Australia"
)

education = (
    "Professional degree",
    "Master’s degree",
    "Less than a Bachelore",
    "Bachelor’s degree"
)

country = st.selectbox("Country", countries)
education = st.selectbox("Education level", education)
expericence = st.slider("Years of Experience", 0, 50, 3)

ok = st.button("Calculate Salary")

if ok:
    # กำหนดค่าให้กับ X
    X = np.array([[country, education, expericence]])
    
    # แปลงข้อมูล categorical (country, education) เป็นค่าตัวเลขโดยใช้ LabelEncoder
    X[:, 0] = le_country.transform(X[:, 0])
    X[:, 1] = le_education.transform(X[:, 1])
    
    # แปลง X ให้เป็น float
    X = X.astype(float)
    
    # ทำนายค่าเงินเดือน
    salary = regressor.predict(X)
    
    st.subheader(f"The estimated salary is ${salary[0]:.2f}")
