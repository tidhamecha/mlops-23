from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
# we will put all utils here



def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model


def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test =  train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
    return X_train, X_test, X_dev, y_train, y_test, y_dev

# Question 2:
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)    
    return metrics.accuracy_score(y_test, predicted)

def tune_hyper_parameters(train_features, train_labels, dev_features, dev_labels, param_combinations):
    highest_accuracy = -1
    best_model = None
    optimal_gamma_value = None
    optimal_c_value = None

    for param_set in param_combinations:
        current_model = train_model(train_features, train_labels, {'gamma': 0.001, 'C': param_set["c_range"]}, model_type="svm")
        current_accuracy = predict_and_eval(current_model, dev_features, dev_labels)

        if current_accuracy > highest_accuracy:
            highest_accuracy = current_accuracy
            optimal_gamma_value = param_set["gamma"]
            optimal_c_value = param_set["c_range"]
            best_model = current_model
            
    return best_model, highest_accuracy, optimal_gamma_value, optimal_c_value
