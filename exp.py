"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, tune_hyper_parameters
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]

C_ranges = [0.1, 1, 2, 5, 10]

param_combinations = [{"gamma": gamma, "c_range": C} for gamma in gamma_ranges for C in C_ranges]

# 1. Get the dataset
X, y = read_digits()
c = 0
for test_size in [0.1, 0.2, 0.3]:
    for dev_size in [0.1, 0.2, 0.3]:
        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size, dev_size)
        X_train, X_test, X_dev = preprocess_data(X_train), preprocess_data(X_test), preprocess_data(X_dev)
        c += 1
        print("\n\n","="*30,f"BEGINING EXPERIMENT: {c}","="*30)
        best_model, best_accuracy, best_gamma, best_c = tune_hyper_parameters(X_train, y_train, X_dev, y_dev, param_combinations)

        print(f"test_size: [{test_size}]\ndev_size: [{dev_size}]\ntrain_size: [{1-test_size-dev_size}]\noptimal_gamma: [{best_gamma}]\noptimal_c: [{best_c}]\nbest_dev_acc: [{best_accuracy}]")

        # Evaluate on test set
        test_acc = predict_and_eval(best_model, X_test, y_test)
        print(f"test_acc===> [{test_acc}]")

test_acc = predict_and_eval(best_model, X_test, y_test)
print("\n\nTest accuracy: ", test_acc)
