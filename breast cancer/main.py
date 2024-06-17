import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset from sklearn
cancer_dataset = sklearn.datasets.load_breast_cancer()
print(cancer_dataset)

# Create a DataFrame from the dataset
data = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)

# Show the first few rows of the DataFrame
print(data.head())

# Append the target column to the DataFrame
data['diagnosis'] = cancer_dataset.target

# Display the last few rows of the DataFrame
print(data.tail())

# Output the shape of the DataFrame
print(data.shape)

# Get DataFrame information
print(data.info())

# Check for any missing values in the DataFrame
print(data.isnull().sum())

# Display statistical measures of the data
print(data.describe())

# Check the distribution of the target variable
print(data['diagnosis'].value_counts())

# Compute the mean of each feature grouped by the target variable
print(data.groupby('diagnosis').mean())

# Define features (X) and target (Y)
X = data.drop(columns='diagnosis', axis=1)
Y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Initialize and train the Logistic Regression model
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, Y_train)

# Predict on the training set
train_preds = clf.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_preds)
print('Training set accuracy: ', train_accuracy)

# Predict on the testing set
test_preds = clf.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_preds)
print('Testing set accuracy: ', test_accuracy)

# Sample input for prediction
sample_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

# Convert sample data to numpy array
sample_np_array = np.asarray(sample_data)

# Reshape the array for a single prediction
sample_reshaped = sample_np_array.reshape(1, -1)

# Make a prediction
sample_prediction = clf.predict(sample_reshaped)
print(sample_prediction)

# Interpret the prediction
if sample_prediction[0] == 0:
    print('The diagnosis is Malignant')
else:
    print('The diagnosis is Benign')
