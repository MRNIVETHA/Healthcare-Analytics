# Assume you have new data for prediction
# Replace this with your actual input data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
import numpy as np
import pickle
#loading the saved model
loaded_model = pickle.load(open('C:/Users/nithu/Downloads/Nivetha/Heart_failure_prediction/XGBoostModel.sav','rb'))
new_data = np.array([[63, 1, 100, 1, 50, 1, 150000, 1.3, 140, 1,0,10]])

# Make predictions on the new data
input_data_as_numpy_array = np.asarray(new_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#std_data = scaler.transform(input_data_reshaped)
#print(std_data)
prediction = loaded_model.predict(input_data_reshaped)

# Map predicted class label to descriptive output
#class_mapping = {0: "Non-heart patient", 1: "Heart patient"}
#predicted_class = class_mapping[predictions[0]]
print(prediction)
if(prediction[0] == 0):
  print('The patient is not a heart-patient')
else:
  print('The patient is a heart-patient ')  
# Print the predicted class
#print("Prediction:", predicted_class)