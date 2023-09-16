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

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams

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

test_sizes =  [0.1, 0.2, 0.3, 0.45]
dev_sizes  =  [0.1, 0.2, 0.3, 0.45]
for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1- test_size - dev_size
        # 3. Data splitting -- to create train and test sets                
        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
        # 4. Data preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)
    
        best_hparams, best_model, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)

        test_acc = predict_and_eval(best_model, X_test, y_test)
        train_acc = predict_and_eval(best_model, X_train, y_train)
        dev_acc = best_accuracy

        print("test_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(test_size, dev_size, train_size, train_acc, dev_acc, test_acc))

