# Cifar Classifier

## Experiment

Changing the loss function for a given a known good convolutional neural network and evaluating it's performance.

## Setup

For setting up a new AWS g2.x2 node running ubuntu 14.04, assuming CUDA already linked

```
virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install keras
pip install Pillow
pip install matplotlib
python cifar_classifier.py
```

## Main pieces of interest

* `cifar_classifier.py` : Contains all of the experiment code including loading data, the model, data generation

* `plot*.py` :  Matplotlib wrapper + keras callback for generating accuracy and loss plots per epoch of training

## Libraries used

* Keras
* Theano
* Matplotlib
* Pillow

Experiments ran on AWS g2.x2 node with K520 GPU with ~270 seconds per epoch.

