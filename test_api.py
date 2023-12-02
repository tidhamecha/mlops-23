from api.app import app
import pytest, json
from utils import read_digits, preprocess_data


def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/", json={"suffix":suffix})
    assert response.status_code == 200    
    assert response.get_json()['op'] == "Hello, World POST "+suffix

def test_post_predict():
    samples = get_samples_each_class()
    i=0
    X, y = read_digits()
    sample = preprocess_data(X[y==i])[0]
    response = app.test_client().post("/predict/svm", json={"image":[sample.tolist()]})    
    assert response.status_code == 200    
    assert response.get_json()['op'] == i
        


    