"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations

# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
# 2.1. SVM
gamma_list = [0.001, 0.01, 0.1, 1]
C_list = [1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list

h_params_combinations = get_hyperparameter_combinations(h_params)
print("h_params_combinations ", len(h_params_combinations))

# 3. Data splitting -- to create train and test sets

# X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)
X_train, X_test, y_train, y_test, X_dev, y_dev = train_test_dev_split(X, y, test_size=0.3, dev_size=0.2)
# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

for h_params in h_params_combinations:
    # 5. Model training
    model = train_model(X_train, y_train, h_params, model_type="svm")
    # Predict the value of the digit on the test subset
    predict_and_eval(model, X_test, y_test)    
    predict_and_eval(model, X_dev, y_dev)


