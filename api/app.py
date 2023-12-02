from flask import Flask, request
from joblib import dump, load
from markupsafe import escape


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

def load_model():
    models={}
    models['svm'] = load("./models/svm_gamma:1_C:10.joblib")
    models['tree'] = load("./models/tree_max_depth:10.joblib")
    models['lr'] = load("./models/lr_saga_solver:saga.joblib")

    return models

models = load_model()
default_model_type = 'svm'
model = models[default_model_type]

@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):        
    model_type = escape(model_type)
    image = request.json["image"]    
    digit = models[model_type].predict(image)    
    return {"op" : int(digit[0])}