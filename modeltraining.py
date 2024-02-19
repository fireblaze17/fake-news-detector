import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

# Load the training data using joblib
X_train = joblib.load('D:/Ankush Python Projects/Fake news detector/Xtrain_sparse.pkl')
Y_train = joblib.load('D:/Ankush Python Projects/Fake news detector/Ytrain.pkl')

# Initialize the SGDClassifier
model = SGDClassifier(loss='log')  # Use loss='log' for logistic regression

# Train the model
model.fit(X_train, Y_train)

# Evaluation on training data
Y_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, Y_train_prediction)
print("Accuracy score on the training data is:", training_data_accuracy)

# Load the testing data using joblib
X_test = joblib.load('D:/Ankush Python Projects/Fake news detector/Xtest_sparse.pkl')
Y_test = joblib.load('D:/Ankush Python Projects/Fake news detector/Ytest.pkl')

# Evaluation on testing data
Y_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, Y_test_prediction)
print("Accuracy score on the testing data is:", testing_data_accuracy)

# Save the trained model
joblib.dump(model, 'D:/Ankush Python Projects/Fake news detector/predictionmodel_sgd.pkl')
print("Done!")
