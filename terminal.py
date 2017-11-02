# Copyright 2017, Kristofer Krus

################################################################################
# IMPORTS
################################################################################


import os
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.objectives import categorical_crossentropy
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy as accuracy
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from enum import Enum


################################################################################
# OPTIONS
################################################################################


# The types of classifiers
class ClassifierType(Enum):
    Linear = 1  # Neural network with no hidden layer
    NNFCL = 2  # NNFCL = Neural Network with only Fully-Connected Layers
    CNN = 3  # CNN = Convolutional Neural Network


# Class for creating initially empty containers
class VariablGroup:
    pass


################################################################################
# OPTIONS
################################################################################


# Neural network type
#classifier_type = ClassifierType.Linear
#classifier_type = ClassifierType.NNFCL
classifier_type = ClassifierType.CNN

# Training options
validation_split = 1/12  # How big proportion of the training + validation data should be used for validation
num_epochs = 20  # For how many epochs we should train the network. Each epoch consists of a number of batches.
batch_size = 50  # How many examples should be processed in parallel on the GPU.
learning_rate = 0.01  # Scale factor for determining the step length in gradient descent. Should be positive.
learning_rate_decay = 0.0  # If positive, the learning rate will decay towards zero throughout the training.
momentum = 0.8  # A kind of "inertia" in the gradient descent algorithm. 0 means no inertia. 1 means infinite inertia.
nesterov = True  # Whether we should use Nesterov momentum (otherwise we will use "ordinary" momentum).
verbose = 1  # Whether Keras should be verbose during training
loss = categorical_crossentropy  # Which loss function to use for the training
optimizer = SGD(lr=learning_rate, momentum=momentum, decay=learning_rate_decay, nesterov=nesterov)  # Which strategy to use to optimize the network
metrics = [accuracy]  # The list metrics to be evaluated and printed during training


################################################################################
# CONSTANTS
################################################################################


num_classes = 10
img_rows, img_cols = 28, 28
input_shape = ((1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1))


################################################################################
# FUNCTIONS
################################################################################


# This function returns the MNIST dataset
def get_data():
    # Get dataset
    data = VariablGroup()
    ((data.train_and_val_images, data.train_and_val_labels),
     (data.test_images, data.test_labels)) = mnist.load_data()

    # Reshape image data
    images_shape = (-1,) + input_shape
    data.train_and_val_images = data.train_and_val_images.reshape(images_shape).astype('float32') / 255
    data.test_images = data.test_images.reshape(images_shape).astype('float32') / 255

    # Reshape label data to one-hot vectors
    data.train_and_val_labels = to_categorical(data.train_and_val_labels, num_classes)
    data.test_labels = to_categorical(data.test_labels, num_classes)

    return data


def normalize_input_data(data):
    # Calculate the average and the standard deviation of the image intensity
    image_mean = np.mean(data.train_and_val_images)
    image_std = np.std(data.train_and_val_images)
    # Scale and offset the input values so that the are zero-centered and have the standard deviation 1
    data.train_and_val_images = (data.train_and_val_images - image_mean) / image_std
    data.test_images = (data.test_images - image_mean) / image_std


# Class for creating, training and deploying a neural network
def main():
    # Construct the network
    if classifier_type == ClassifierType.Linear:
        # Create a linear classifier
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(num_classes, activation='softmax'))
    elif classifier_type == ClassifierType.NNFCL:
        # Create a neural network with only fully-connected layers
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    elif classifier_type == ClassifierType.CNN:
        # Create a convolutional neural network
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))

    # Configure the model for training
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)

    # Get the data to train on
    data = get_data()
    normalize_input_data(data)

    # Train model
    history = model.fit(
        x=data.train_and_val_images,
        y=data.train_and_val_labels,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=verbose,
        validation_split=validation_split,
        validation_data=None)


if __name__ == '__main__':
    main()
