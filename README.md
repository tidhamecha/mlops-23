# mlops-23
for the mlops course at iitj 2023

dummy commit to make the feature branch ahead of main branch.

e.g. n=100 samples, 2-class/binary classification: image of carrot or turnip.
    50 samples : carrots
    50 sampes : turnips 
        Data distribution: balanced/uniform

    x amount of data for training
    n-x amount of data for testing

    calcuate some eval metric(train model (70samples in training: 35 carrots, 35 turnips), 30samples in testing: 15, 15) == performance

In practise:
    train, dev, test

    train = training the model (model type, model hyper params, model iteration)
    dev = selecting the model
    test = reporting the performance









system requirements:
OS 
h/w -- may be skipped -- general commodity h/w is required

how to setup:
install conda

conda create -n digits python=3.9
conda activate digits
pip install -r requirements.txt

how to run

python exp.py


Meaning of failure:
- poor performance metrics
- coding runtime/compile error


- the model gave bad predictions on the new test samples during demo.

feature
- vary model hyper parameters
- 