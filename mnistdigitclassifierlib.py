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
from numpy.random import randint
from enum import Enum


################################################################################
# CLASSES
################################################################################


# Class for creating initially empty containers
class VariablGroup:
    pass


# The types of classifiers
class ClassifierType(Enum):
    Linear = 1
    NNFCL = 2  # NNFCL = neural network with only fully-connected layers
    CNN = 3  # CNN = convolutional neural network


# Class for creating, training and deploying a neural network
class MnistDigitClassifier:
    # Constants
    num_classes = 10
    img_rows, img_cols = 28, 28

    # Constructor
    def __init__(self,
                 classifier_type=ClassifierType.CNN,
                 model_file=None,
                 batch_normalization=True,
                 dropout=True,
                 validation_split=1/12):
        
        # Derived values
        self._input_shape = ((1, self.img_rows, self.img_cols) if K.image_data_format() == 'channels_first' else
                             (self.img_rows, self.img_cols, 1))

        # Load or create model
        self._model_file = model_file if isinstance(model_file, str) else None
        if isinstance(self._model_file, str) and os.path.isfile(self._model_file):
            self._model = load_model(self._model_file)
        else:
            if classifier_type == ClassifierType.Linear:
                # Create a linear classifier
                self._model = Sequential()
                self._model.add(Flatten(input_shape=self._input_shape))
                if dropout:
                    self._model.add(Dropout(0.5))
                self._model.add(Dense(self.num_classes, activation='softmax'))
            elif classifier_type == ClassifierType.NNFCL:
                # Create a neural network with only fully-connected layers
                self._model = Sequential()
                self._model.add(Flatten(input_shape=self._input_shape))
                if dropout:
                    self._model.add(Dropout(0.5))
                self._model.add(Dense(512, activation='relu'))
                if dropout:
                    self._model.add(Dropout(0.5))
                if batch_normalization:
                    self._model.add(BatchNormalization())
                self._model.add(Dense(512, activation='relu'))
                if dropout:
                    self._model.add(Dropout(0.5))
                if batch_normalization:
                    self._model.add(BatchNormalization())
                self._model.add(Dense(self.num_classes, activation='softmax'))
            elif classifier_type == ClassifierType.CNN:
                # Create a convolutional neural network
                self._model = Sequential()
                self._model.add(InputLayer(input_shape=self._input_shape))
                self._model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                if batch_normalization:
                    self._model.add(BatchNormalization())
                self._model.add(Conv2D(64, (3, 3), activation='relu'))
                self._model.add(MaxPooling2D(pool_size=(2, 2)))
                self._model.add(Flatten())
                if dropout:
                    self._model.add(Dropout(0.5))
                if batch_normalization:
                    self._model.add(BatchNormalization())
                self._model.add(Dense(128, activation='relu'))
                if dropout:
                    self._model.add(Dropout(0.5))
                if batch_normalization:
                    self._model.add(BatchNormalization())
                self._model.add(Dense(self.num_classes, activation='softmax'))

        self._validation_split = validation_split
        self._data = None
        self._image_mean = None
        self._image_std = None

    # Train the network
    def train(self, batch_size=50, num_epochs=20, learning_rate=0.2, learning_rate_decay=0.0, momentum=0.0, nesterov=False, verbose=1):

        loss = categorical_crossentropy
        optimizer = SGD(lr=learning_rate, momentum=momentum, decay=learning_rate_decay, nesterov=nesterov)
        metrics = [accuracy]

        # Configure the model for training
        self._model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)

        # Train model
        history = self._model.fit(
            x=self.normalized_input(self.data.train_and_val_images),
            y=self.data.train_and_val_labels,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=verbose,
            validation_split=self._validation_split,
            validation_data=None)

        # If a model file name is give, save the resulting trained network
        if self._model_file:
            self._model.save(self._model_file)

        # Finally, print the training result
        print("Training history:")
        for key, val in history.history.items():
            print("    {}: {}".format(key, val))

    def normalized_input(self, input):
        return (input - self.image_mean) / self.image_std

    # Predict the class for an individual image
    def predict(self, image, batch_size=32, verbose=0):
        input = self.normalized_input(image.reshape((-1,) + self._input_shape).astype('float32') / 255)
        return self._model.predict(input, batch_size=batch_size, verbose=verbose)[0]

    def get_random_test_data_example(self):
        idx = randint(0, self.data.test_images.shape[0])
        return self.data.test_images[idx], self.data.test_labels[idx]

    # This property holds the MNIST dataset
    @property
    def data(self):
        if self._data is None:
            # Get dataset
            self._data = VariablGroup()
            ((self._data.train_and_val_images, self._data.train_and_val_labels),
             (self._data.test_images         , self._data.test_labels         )) = mnist.load_data()

            # Reshape image data
            images_shape = (-1,) + self._input_shape
            self._data.train_and_val_images = self._data.train_and_val_images.reshape(images_shape).astype('float32') / 255
            self._data.test_images          = self._data.test_images         .reshape(images_shape).astype('float32') / 255

            # Reshape label data to one-hot vectors
            self._data.train_and_val_labels = to_categorical(self._data.train_and_val_labels, self.num_classes)
            self._data.test_labels          = to_categorical(self._data.test_labels         , self.num_classes)
        return self._data

    # This property holds the mean intensity of the images in the MNIST dataset
    @property
    def image_mean(self):
        if self._image_mean is None:
            self._image_mean = np.mean(self.data.train_and_val_images)
        return self._image_mean

    # This property holds the standard deviation of the  intensity of the images in the MNIST dataset
    @property
    def image_std(self):
        if self._image_std is None:
            self._image_std = np.std(self.data.train_and_val_images)
        return self._image_std