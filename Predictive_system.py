# Assume you have new data for prediction
# Replace this with your actual input data

from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
#loading the model
loaded_model = pickle.load(open('C:/Users/nithu/Downloads/Nivetha/Heart_failure_prediction/TrainedXGBoostModel.sav','rb'))
new_data = np.array([[55, 0, 7861, 0, 38, 0, 263358,1.1, 136,1, 0,0]])
#standardized_data = scaler.transform(new_data)
# Make predictions on the new data
input_data_as_numpy_array = np.asarray(new_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
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
#print("Prediction:", predicted_class)'''
