from sklearn.model_selection import train_test_split
from sklearn import svm, tree, datasets, metrics, preprocessing
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
# we will put all utils here

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

def tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations, model_type="svm"):
    best_accuracy = -1
    best_model_path = ""
    for h_params in h_params_combinations:
        # 5. Model training
        model = train_model(X_train, y_train, h_params, model_type=model_type)
        # Predict the value of the digit on the test subset        
        cur_accuracy, _, _ = predict_and_eval(model, X_dev, y_dev)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_hparams = h_params
            best_model_path = "./models/{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in h_params.items()]) + ".joblib"
            best_model = model

    # save the best_model    
    dump(best_model, best_model_path) 


    return best_hparams, best_model_path, best_accuracy 



def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    data = preprocessing.normalize(data, norm='l2', axis=1)
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, shuffle = True
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    if model_type == "tree":
        # Create a classifier: a decision tree classifier
        clf = tree.DecisionTreeClassifier
    
    if model_type == "lr_lbfgs" or model_type == "lr_newton-cg" or model_type == "lr_saga" or model_type == "lr_sag":
        # Create a classifier: a decision tree classifier
        clf = LogisticRegression
            
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model


def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test =  split_data(X, y, test_size=test_size, random_state=1)
    print("train+dev = {} test = {}".format(len(Y_train_Dev),len(y_test)))
    
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
        
    return X_train, X_test, X_dev, y_train, y_test, y_dev

# Question 2:
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted), metrics.f1_score(y_test, predicted, average="macro"), predicted

