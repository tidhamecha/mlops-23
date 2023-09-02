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
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train and test sets

# X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)
X_train, X_test, y_train, y_test, X_dev, y_dev = train_test_dev_split(X, y, test_size=0.3, dev_size=0.2)
# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

# 6. Getting model predictions on test set
# Predict the value of the digit on the test subset
# 7. Qualitative sanity check of the predictions
# 8. Evaluation
predict_and_eval(model, X_test, y_test)
