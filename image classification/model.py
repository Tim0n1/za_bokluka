import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1,)
classes = ['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog','horse','ship','truck']
def plot_sample(X, y, index):
    plt.figure(figsize=(15,2), dpi=100)
    plt.imshow(X[index])
    plt.ylabel(classes[y[index]])
    plt.show()

plot_sample(X_train, y_train, 0)
