# Basic-Face-Detection
A simple Face Detection system using a Deep Neural Network

## Included packages
- numpy
- scikit-learn
- scipy
- matplotlib

##### Notice: Input training set dimensions in this project is `(n, m)`, which `m` is number of samples and `n` in the number of features. But the input training set dimensions is `(m, 1)`.

## About Classifier
This classifier is training on olivetti faces dataset, 400 face images with size `64*64`. The Neural Network is equipped to DropOut regularization and L2 regularization techniques. In the last part of the notebook you can see a classifier added from scikit-learn package to double check the model you have trained.
