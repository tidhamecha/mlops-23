from api.app import app
import pytest
from sklearn.datasets import fetch_openml
import numpy as np
import random

  

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
    # Assuming you have a function to fetch a sample for each digit from the dataset
    # Replace `get_sample_for_digit` with your actual implementation
    for digit in range(10):
        sample = get_sample_for_digit(digit)
        response = app.test_client().post("/predict", json={"image": sample})
        assert response.status_code == 200
        assert response.get_data() == str(digit).encode()
        



def get_sample_for_digit(digit):
    # Find indices of images with the specified digit
    
    mnist = fetch_openml(name="mnist_784", version=1)
    images, labels = mnist.data, mnist.target.astype(int)
    images = images.values.reshape((-1, 28, 28))
    digit_indices = np.where(labels == digit)

    # Randomly select one index
    random_index = random.choice(digit_indices[0])

    # Get the corresponding image
    sample_image = images[random_index]

    # Flatten the 2D image to a 1D array
    flattened_sample = sample_image.flatten()

    return flattened_sample
        
        
