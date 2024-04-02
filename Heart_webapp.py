import pandas as pd
import numpy as np
import streamlit as st
import pickle

def main():
    st.set_page_config(
        page_title="Heart Failure Prediction",
        page_icon="❤️",
        layout="wide"
    )

    # Load the trained model
    #filename = 'C:/Users/nithu/Downloads/Nivetha/Heart_failure_prediction/TrainedXGBoostModel2.sav'
    
    def load_model():
     model_path = os.path.join(os.path.dirname(__file__), 'TrainedXGBoostModel2.sav')
     with open(model_path, 'rb') as file:
        model = pickle.load(file)
     return model
    st.markdown(
        """
        <style>
            .main {
                text-align: center;
            }
            h1 {
                font-size: 36px;
                color: #B72F39;
            }
            h2 {
                font-size: 24px;
                color: #333333;
            }
            h3 {
                font-size: 18px;
                color: #333333;
            }
            p {
                font-size: 16px;
                color: #555555;
            }
            .btn {
                background-color: #B72F39;
                color: #FFFFFF;
                font-size: 18px;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .btn:hover {
                background-color: #971F2C;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Heart Failure Prediction")

    # Input form
    st.sidebar.title("Enter Patient Details")
    age = st.sidebar.slider("Age", 0, 150, 30)
    anaemia = st.sidebar.radio("Anaemia", ['No', 'Yes'])
    creatinine_phosphokinase = st.sidebar.number_input("Creatinine Phosphokinase", min_value=0)
    diabetes = st.sidebar.radio("Diabetes", ['No', 'Yes'])
    ejection_fraction = st.sidebar.slider("Ejection Fraction", 0, 100, 50)
    high_blood_pressure = st.sidebar.radio("High Blood Pressure", ['No', 'Yes'])
    platelets = st.sidebar.number_input("Platelets", min_value=0)
    serum_creatinine = st.sidebar.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.sidebar.number_input("Serum Sodium", min_value=0)
    sex = st.sidebar.radio("Gender", ['Female', 'Male'])
    smoking = st.sidebar.radio("Smoking", ['No', 'Yes'])
    time = st.sidebar.number_input("Time", min_value=0)

    if st.sidebar.button("Predict"):
        # Preprocess input data
        anaemia_val = 1 if anaemia == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        high_blood_pressure_val = 1 if high_blood_pressure == "Yes" else 0
        sex_val = 1 if sex == "Male" else 0
        smoking_val = 1 if smoking == "Yes" else 0

        # Make prediction
        prediction = model.predict([[age, anaemia_val, creatinine_phosphokinase, diabetes_val, ejection_fraction, high_blood_pressure_val, platelets, serum_creatinine, serum_sodium, sex_val, smoking_val, time]])

        if prediction[0] == 0:
            st.error("The patient is at high risk of heart failure.")
        else:
            st.success("The patient is not at risk of heart failure.")

if __name__ == "__main__":
    main()
