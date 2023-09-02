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
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train and test sets

X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=0.3, dev_size=0.2)
# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

# HYPER PARAMETER TUNING
# - take all combinations of gamma and C
best_acc_so_far = -1
best_model = None


for cur_gamma in gamma_ranges:
    for cur_C in C_ranges:
        # print("Running for gamma={} C={}".format(cur_gamma, cur_C))
        # - train model with cur_gamma and cur_C
        # # 5. Model training
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_C}, model_type="svm")
        # - get some performance metric on DEV set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        # - select the hparams that yields the best performance on DEV set
        if cur_accuracy > best_acc_so_far:
            print("New best accuracy: ", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_C = cur_C
            best_model = cur_model
print("Optimal parameters gamma: ", optimal_gamma, "C: ", optimal_C)


# 6. Getting model predictions on test set
# 7. Qualitative sanity check of the predictions
# 8. Evaluation
test_acc = predict_and_eval(best_model, X_test, y_test)
print("Test accuracy: ", test_acc)
