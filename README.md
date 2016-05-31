# Cifar Classifier

# Experiment

Changing the loss function for a given a known good convolutional neural network and evaluating it's performance.

# Main pieces of interest

* `cifar_classifier.py` : Contains all of the experiment code including loading data, the model, data generation

* `plot*.py` :  Matplotlib wrapper + keras callback for generating accuracy and loss plots per epoch of training

# Libraries used

* Keras
* Theano
* Matplotlib
* Pillow

Experiments ran on AWS g2.x2 node with K520 GPU with ~270 seconds per epoch.