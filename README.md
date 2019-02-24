[//]: # (Atom: Turn on preview using ctrl+shift+m)

This is a simple demonstrator of an [MNIST](http://yann.lecun.com/exdb/mnist/) classifier, with a graphical user interface and a canvas that lets you draw digits and have them classified in real-time.

## Prerequisites

* [Python 3](https://www.python.org/) (3.6.8; make sure the version is compatible TensorFlow)
* [TensorFlow](https://www.tensorflow.org/install) (tensorflow-gpu 1.12.0)
* [Keras](https://keras.io/) (2.2.4)
* [PySide 2](https://pypi.org/project/PySide2/) (5.12.1)
* [h5py](https://pypi.org/project/h5py/) (2.9.0)

The text within parenthesis specifies the versions this repository is known to be compatible with.

## Usage

Run the graphical user interface as follows:

    $ python gui.py

Draw with the left mouse button and reset the canvas with the right mouse button.

The easiest way to tweak the behavior of the application is currently to change the hardcoded values for the parameters under the options section in `gui.py`, as the script currently doesn't take any arguments.

The first time the script is run with a specific model type selected, a new model will be trained from scratch and saved as an HDF5 file in the folder from which the script is run. This file will then be loaded and used the following times the application is run with the same model type selected instead of training a new model.

## License

[GNU General Public License v3.0](COPYING)
