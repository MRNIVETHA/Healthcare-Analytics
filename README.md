### Heart Failure Prediction ###

---

### Overview:

This project focuses on predicting heart failure using machine learning techniques. Heart failure is a critical medical condition that requires timely detection and intervention to prevent adverse outcomes. By leveraging various features such as medical history, vital signs, and demographic information, this predictive model aims to assist healthcare professionals in identifying individuals at risk of heart failure.

### Dataset:

The dataset used for this project consists of anonymized patient data collected from various healthcare facilities. It includes both numerical and categorical features such as age, sex, blood pressure, serum creatinine levels, ejection fraction, and more. Additionally, the dataset contains a target variable indicating whether a patient experienced heart failure or not.

### Features:

1. Age: Age of the patient.
2. Sex: Gender of the patient (Male/Female).
3. Blood Pressure: Systolic blood pressure.
4. Serum Creatinine: Level of serum creatinine in the blood.
5. Ejection Fraction: Percentage of blood leaving the heart at each contraction.
6. Diabetes: Whether the patient has diabetes or not.
7. Smoking: Whether the patient smokes or not.
8. Anaemia: Whether the patient has anaemia or not.
9. High Blood Pressure: Whether the patient has high blood pressure or not.

### Methodology:

1. **Data Preprocessing**: Missing values handling, feature scaling, and encoding categorical variables.
2. **Feature Selection**: Identifying significant features using techniques like correlation analysis or feature importance.
3. **Model Selection**: Evaluating various machine learning algorithms such as logistic regression, random forest, and gradient boosting to determine the best performing model.
4. **Model Training**: Training the selected model on the dataset.
5. **Model Evaluation**: Assessing the model's performance using metrics like accuracy, precision, recall, and F1-score.
6. **Hyperparameter Tuning**: Optimizing model parameters to improve performance.
7. **Validation**: Validating the model on unseen data to ensure generalization.
8. **Deployment**: Deploying the model for real-time heart failure prediction.

### Results:
The XGBoost model achieved satisfactory performance with an accuracy of 88% on the validation dataset. The precision for predicting absence of heart failure (0.0) was 89%, and for predicting presence of heart failure (1.0) was 88%. The recall for predicting absence of heart failure (0.0) was 95%, and for predicting presence of heart failure (1.0) was 74%. The F1-score for predicting absence of heart failure (0.0) was 92%, and for predicting presence of heart failure (1.0) was 80%.

This indicates a reasonably good performance of the model in distinguishing between individuals with and without heart failure. 
Further fine-tuning and optimization may lead to enhanced predictive capabilities.

### Usage:

1. **Data Preparation**: Ensure the input data is in the same format as the training dataset.
2. **Model Loading**: Load the trained model.
3. **Prediction**: Input patient data into the model to predict the likelihood of heart failure.
4. **Interpretation**: Interpret the model's output and take appropriate actions based on the predicted risk.

### Dependencies:

- Python 3.12
- Scikit-learn
- Pandas
- NumPy

---
