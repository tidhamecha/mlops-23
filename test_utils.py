from utils import get_hyperparameter_combinations, train_test_dev_split,read_digits, tune_hparams, preprocess_data
from joblib import load
import sklearn
import os
def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

def create_dummy_hyperparameter():
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    return h_params_combinations
def create_dummy_data():
    X, y = read_digits()
    
    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev = X[:50,:,:]
    y_dev = y[:50]

    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)

    return X_train, y_train, X_dev, y_dev
def test_for_hparam_cominations_values():    
    h_params_combinations = create_dummy_hyperparameter()
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

def test_model_saving():
    X_train, y_train, X_dev, y_dev = create_dummy_data()
    h_params_combinations = create_dummy_hyperparameter()

    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)   

    assert os.path.exists(best_model_path)

def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .1
    dev_size = .6

    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert  ((len(X_dev) == 60))

def test_lr_model():
    model_path = "./models/lr_sag_solver:sag.joblib"
    model_loaded = load(model_path) 
    assert type(model_loaded) == sklearn.linear_model.LogisticRegression
