import os
import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier


current_dir = os.path.dirname(os.path.abspath(__file__))


X_train_path = os.path.join(current_dir, 'Xtrain_sparse.pkl')
Y_train_path = os.path.join(current_dir, 'Ytrain.pkl')
X_test_path = os.path.join(current_dir, 'Xtest_sparse.pkl')
Y_test_path = os.path.join(current_dir, 'Ytest.pkl')
model_path = os.path.join(current_dir, 'predictionmodel_sgd.pkl')


X_train = joblib.load(X_train_path)
Y_train = joblib.load(Y_train_path)

# Initialize the SGDClassifier
model = SGDClassifier(loss='log_loss') 

# Train the model
model.fit(X_train, Y_train)

# Evaluation on training data
Y_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, Y_train_prediction)
print("Accuracy score on the training data is:", training_data_accuracy)

# Load the testing data using joblib
X_test = joblib.load(X_test_path)
Y_test = joblib.load(Y_test_path)

# Evaluation on testing data
Y_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, Y_test_prediction)
print("Accuracy score on the testing data is:", testing_data_accuracy)

model_path = os.path.join(current_dir, "predictionmodel_sgd.pkl")
joblib.dump(model, model_path)


