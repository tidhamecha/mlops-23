from utils import get_hyperparameter_combinations, train_test_dev_split,read_digits

def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

def test_for_hparam_cominations_values():    
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)


def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .1
    dev_size = .6
    train_size = 1 - test_size - dev_size

    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert  ((len(X_dev) == 60))