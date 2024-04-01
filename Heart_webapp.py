import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('C:/Users/nithu/Downloads/Nivetha/Heart_failure_prediction/XGBoostModel.sav','rb'))
def heart_prediction(new_data):
#new_data = np.array([[63, 1, 100, 1, 50, 1, 150000, 1.3, 140, 1,0,10]])

# Make predictions on the new data
    input_data_as_numpy_array = np.asarray(new_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#std_data = scaler.transform(input_data_reshaped)
#print(std_data)
    prediction = loaded_model.predict(input_data_reshaped)


    print(prediction)

    if(prediction[0] == 0):
       return 'non_heart'
    else:
       return 'heart'
def main():

    st.title('Heart Failure Prediction Web App') 

    # Getting the input 
    age = st.number_input('Age of the person', min_value=0, max_value=150)
    anaemia = st.number_input('Whether the person is anaemic', min_value=0, max_value=1)
    creatinine_phosphokinase = st.number_input('Level of creatinine phosphokinase', min_value=0)
    diabetes = st.number_input('Whether the person is diabetic', min_value=0, max_value=1)
    ejection_fraction = st.number_input('Ejection fraction of a person', min_value=0, max_value=100)
    high_blood_pressure = st.number_input('High Blood pressure', min_value=0, max_value=1)
    platelets = st.number_input('Platelet count', min_value=0)
    serum_creatinine = st.number_input('Level of serum creatinine', min_value=0.0)
    serum_sodium = st.number_input('Level of serum sodium', min_value=0)
    sex = st.number_input('Gender (0 for female, 1 for male)', min_value=0, max_value=1)
    smoking = st.number_input('Whether the person is a smoker', min_value=0, max_value=1)
    time = st.number_input('Time', min_value=0)

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Prediction Test Results'):
        new_data = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]
        diagnosis = heart_prediction(new_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()    



