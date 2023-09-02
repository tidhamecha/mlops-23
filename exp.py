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
from utils import preprocess_data, split_data, train_model, read_digits

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train and test sets

X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)

# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

# 6. Getting model predictions on test set
# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

# 7. Qualitative sanity check of the predictions

# 8. Evaluation
print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
