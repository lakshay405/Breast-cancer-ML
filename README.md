# Breast-cancer-ML
Breast Cancer Classification Project
This project utilizes the breast cancer dataset from sklearn to build a classification model using Logistic Regression.

Data Exploration and Preprocessing
The breast cancer dataset is loaded from sklearn.
Features are loaded into a DataFrame (data) and the target ('diagnosis') is appended.
Basic data exploration includes:
Printing dataset details and shape.
Checking for missing values and displaying statistical summaries.
Exploring the distribution of the target variable ('diagnosis').
Model Building and Evaluation
Logistic Regression Model:
Features (X) are defined by dropping the 'diagnosis' column.
Target (Y) is set to 'diagnosis'.
Data is split into training and testing sets using train_test_split.
Logistic Regression model is initialized and trained on the training data.
Model performance is evaluated on both training and testing sets using accuracy score.
Prediction
A sample data point is defined for prediction.
The sample data is converted to a numpy array and reshaped for single prediction.
The trained model predicts the diagnosis ('Malignant' or 'Benign') based on the sample data.
Results
The accuracy of the Logistic Regression model is printed for both the training and testing sets.
The predicted diagnosis for the sample data is displayed along with an interpretation.
