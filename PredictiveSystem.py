from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
#loading the model
loaded_model = pickle.load(open('C:/Users/nithu/Downloads/Nivetha/coapps/XGBModel.sav','rb'))
input_data = np.array([[55, 0, 7861, 0, 38, 0, 263358,1.1, 136,1, 0,0]])
prediction = loaded_model.predict(input_data)
print(prediction)
if(prediction[0] == 0):
  print('The patient is at low risk of heart failure.')
else:
  print('The patient is at high risk of heart failure.')  
