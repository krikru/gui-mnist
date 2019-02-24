# Copyright 2017, 2019 Kristofer Krus
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <https://www.gnu.org/licenses/>.


################################################################################
# IMPORTS
################################################################################


import os
import sys
from PySide2.QtCore import Qt, Signal, Slot, QRect
from PySide2.QtGui import QColor, QImage, QPen, QPainter, QPainterPath
from PySide2.QtWidgets import QWidget, QApplication, QMainWindow, QLayout, QVBoxLayout, QFormLayout, QLabel, QPushButton
import numpy as np
from mnistdigitclassifierlib import ClassifierType, MnistDigitClassifier


################################################################################
# OPTIONS
################################################################################


# Application
#classify_images = False
classify_images = True
#skip_training_if_pretrained_model_exists = False
skip_training_if_pretrained_model_exists = True

# Neural network type
#classifier_type = ClassifierType.Linear  # Neural network with no hidden layer
#classifier_type = ClassifierType.NNFCL  # NNFCL = Neural Network with only Fully-Connected Layers
classifier_type = ClassifierType.CNN  # CNN = Convolutional Neural Network

# Neural network training
num_epochs = 20  # For how many epochs we should train the network. Each epoch consists of a number of batches.
batch_size = 50  # How many examples should be processed in parallel on the GPU.
learning_rate = 0.01  # Scale factor for determining the step length in gradient descent. Should be positive.
learning_rate_decay = 0.0  # If positive, the learning rate will slowly decay towards zero throughout the training.
momentum = 0.8  # A kind of "inertia" in the gradient descent algorithm. 0 means no inertia. 1 means infinite inertia.
nesterov = True  # Whether we should use Nesterov momentum (otherwise we will use "ordinary" momentum).
batch_normalization = True  # Whether batch normalization should be used
dropout = True  # Whether dropout should be used
verbose = 1  # Whether Keras should be verbose during training

# File names for storage of training progress.
model_file = {ClassifierType.Linear: 'model-linear.h5',
              ClassifierType.NNFCL : 'model-nnfcl.h5' ,
              ClassifierType.CNN   : 'model-cnn.h5'   }.get(classifier_type, None)

# Drawing (for drawing digits in the GUI)
pen_width = 2.5  # The width of the pen stroke on the canvas in pixels

# GUI layout
canvas_scale = 10


################################################################################
# CONSTANTS
################################################################################


# Image shape
w, h = MnistDigitClassifier.img_cols, MnistDigitClassifier.img_rows

# Classification task
num_classes = MnistDigitClassifier.num_classes


################################################################################
# CLASSES
################################################################################


class Canvas(QWidget):
    content_changed = Signal()

    _background_color = QColor.fromRgb(0, 0, 0)
    _foreground_color = QColor.fromRgb(255, 255, 255)

    def __init__(self, parent, w, h, pen_width, scale):
        super().__init__(parent)
        self.w = w
        self.h = h
        self.scaled_w = scale * w
        self.scaled_h = scale * h
        self.scale = scale

        # Set size
        self.setFixedSize(self.scaled_w, self.scaled_h)

        # Create image
        self.small_image = QImage(self.w, self.h, QImage.Format_RGB32)
        self.small_image.fill(self._background_color)
        self.large_image = QImage(self.scaled_w, self.scaled_h, QImage.Format_RGB32)
        self.large_image.fill(self._background_color)

        # Create pen
        self.pen = QPen()
        self.pen.setColor(self._foreground_color)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setWidthF(scale * pen_width)

        # There is currently no path
        self.currentPath = None

        self.content_changed.connect(self.repaint)

    def _get_painter(self, paintee):
        painter = QPainter(paintee)
        painter.setPen(self.pen)
        painter.setRenderHint(QPainter.Antialiasing, True)
        return painter

    def _derive_small_image(self, large_image=None):
        if large_image is None:
            large_image = self.large_image
        # Downsample image
        self.small_image = large_image.scaled(self.w, self.h, mode=Qt.SmoothTransformation)
        self.content_changed.emit()

    def _current_path_updated(self, terminate_path=False):
        # Determine whether to draw on the large image directly or whether to make a temporary copy
        paintee = self.large_image if terminate_path else self.large_image.copy()

        # Draw path on the large image of choice
        painter = self._get_painter(paintee)
        if self.currentPath.elementCount() != 1:
            painter.drawPath(self.currentPath)
        else:
            painter.drawPoint(self.currentPath.elementAt(0))
        painter.end()

        # Optionally terminate the path
        if terminate_path:
            self.currentPath = None

        # Downsample image
        self._derive_small_image(paintee)

    def _clear_image(self):
        self.large_image.fill(self._background_color)
        self._derive_small_image()

    def get_content(self):
        return np.asarray(self.small_image.constBits()).reshape((self.h, self.w, -1))

    def set_content(self, image_rgb):
        for row in range(image_rgb.shape[0]):
            for col in range(image_rgb.shape[1]):
                self.small_image.setPixel(col, row, image_rgb[row, col])
        self.large_image = self.small_image.scaled(self.scaled_w, self.scaled_h, mode=Qt.SmoothTransformation)
        self._derive_small_image()
        self.content_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Create new path
            self.currentPath = QPainterPath()
            self.currentPath.moveTo(event.pos())
            self._current_path_updated()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.currentPath is not None:
            # Add point to current path
            self.currentPath.lineTo(event.pos())
            self._current_path_updated()

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton) and self.currentPath is not None:
            # Add terminal point to current path
            self.currentPath.lineTo(event.pos())
            self._current_path_updated(terminate_path=True)
        elif event.button() == Qt.RightButton:
            self._clear_image()

    def paintEvent(self, event):
        paint_rect = event.rect()  # Only paint the surface that needs painting
        painter = self._get_painter(self)

        # Draw image
        painter.scale(self.scale, self.scale)
        painter.drawImage(paint_rect, self.small_image, paint_rect)

        painter.end()

        painter = self._get_painter(self)

        #if self.currentPath is not None:
        #    painter.drawPath(self.currentPath)

    @Slot()
    def repaint(self):
        super().repaint()


class BarChartBar(QWidget):
    def __init__(self, parent, value=None, max_value=None, foreground_color=Qt.GlobalColor.red,
                 background_color=Qt.GlobalColor.white, *argv, **kwargs):
        super().__init__(parent, *argv, **kwargs)
        self.foreground_color = QColor(foreground_color)
        self.background_color = QColor(background_color)
        self.set_value(value, max_value)

    def set_value(self, value, max_value):
        self.value     = value
        self.max_value = max_value
        self.repaint()

    def paintEvent(self, event):
        if self.value is None or self.max_value is None:
            return
        painter = QPainter(self)
        f_width = (self.value/self.max_value) * self.width()
        f_width_whole_part = min(int(f_width), self.width()-1)
        f_width_fractional_part = f_width - f_width_whole_part
        b_width_whole_part = self.width() - f_width_whole_part - 1
        middle_color = interpolate_qcolor(self.foreground_color, self.background_color, f_width_fractional_part)
        painter.fillRect(QRect(0, 0, f_width_whole_part, self.height()), self.foreground_color)
        painter.fillRect(QRect(f_width_whole_part, 0, 1, self.height()), middle_color)
        painter.fillRect(QRect(f_width_whole_part + 1, 0, b_width_whole_part, self.height()), self.background_color)


class BarChart(QWidget):
    def __init__(self, parent, num_bars, values=None, max_value=None):
        super().__init__(parent)

        # Create class values
        self.num_bars = num_bars
        self.values = values
        self.max_value = max_value

        if self.values is None:
            self.values = [None] * self.num_bars

        # Create layout and widgets
        layout = QFormLayout()
        self.setLayout(layout)
        self.labels = []
        self.drawing_widgets = []
        for idx in range(self.num_bars):
            label = QLabel(str(idx))
            self.labels.append(label)
            drawing_widget = BarChartBar(self)
            self.drawing_widgets.append(drawing_widget)
            layout.addRow(label, drawing_widget)

        self.update_bars()

    def set_values(self, values, max_value=None):
        self.values = values
        self.max_value = max_value
        self.update_bars()

    def set_label_attributes(self, attribute_dict):
        label_texts = [str(idx) for idx in range(self.num_bars)]
        for idx, attributes in attribute_dict.items():
            for attribute in attributes:
                label_texts[idx] = '<' + attribute + '>' + label_texts[idx] + '</' + attribute + '>'
        for label, label_text in zip(self.labels, label_texts):
            label.setText(label_text)

    def update_bars(self):
        max_value = self.max_value if self.max_value else max([val for val in self.values if val is not None] + [0])
        for idx in range(self.num_bars):
            self.drawing_widgets[idx].set_value(self.values[idx], max_value)


class MnistClassifierDemonstrator(QMainWindow):
    def __init__(self, mnist_digit_classifier=None, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.mnist_digit_classifier = mnist_digit_classifier

        layout = QVBoxLayout()
        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle("MNIST digit classifier")

        # Create a canvas
        self.canvas = Canvas(self, w, h, pen_width, canvas_scale)
        layout.addWidget(self.canvas)
        self.canvas.content_changed.connect(self.on_canvas_content_changed)

        # Create bar chart
        self.bar_chart = BarChart(self, num_classes)
        layout.addWidget(self.bar_chart)

        # Create a button for retrieving a random test image from the dataset
        self.load_random_test_image_button = QPushButton(self)
        self.load_random_test_image_button.setText("Load random test image")
        layout.addWidget(self.load_random_test_image_button)
        self.load_random_test_image_button.clicked.connect(self.on_load_random_test_image_button_clicked)

        # Initialize bar chart state
        self.on_canvas_content_changed()

    @Slot()
    def on_canvas_content_changed(self):
        if self.mnist_digit_classifier:
            gray_scale_content = np.average(self.canvas.get_content()[:, :, :3], axis=2)
            predicted = self.mnist_digit_classifier.predict(gray_scale_content)
            self.bar_chart.set_label_attributes(dict())
            self.bar_chart.set_values(values=predicted, max_value=1)

    @Slot()
    def on_load_random_test_image_button_clicked(self):
        if self.mnist_digit_classifier:
            image, label = self.mnist_digit_classifier.get_random_test_data_example()
            image_rgb = (image.reshape((h, w))*255 + 0.5).astype(int) * 0x010101
            self.canvas.set_content(image_rgb)
            self.bar_chart.set_label_attributes({np.argmax(label): ['b']})
        pass


################################################################################
# FUNCTIONS
################################################################################


def interpolate_qcolor(front, back, alpha):
    [c1, c2] = [np.append(qcolor.alphaF() * np.array([qcolor.redF(), qcolor.greenF(), qcolor.blueF()]), qcolor.alphaF())
                for qcolor in [front, back]]
    c = alpha * c1 + (1 - alpha) * c2
    c[:3] *= 1 / c[3] if c[3] > 0 else 0
    return QColor(*((255 * c).astype(int).tolist()))


def main():
    if classify_images:
        # Create the MnistDigitClassifier object
        mnist_digit_classifier = MnistDigitClassifier(classifier_type=classifier_type,
                                                      model_file=model_file,
                                                      batch_normalization=batch_normalization,
                                                      dropout=dropout)
        # Determine whether to train the model
        model_file_exists = isinstance(model_file, str) and os.path.isfile(model_file)
        train_model = not (skip_training_if_pretrained_model_exists and model_file_exists)
        if train_model:
            # Train model
            mnist_digit_classifier.train(batch_size=batch_size,
                                         num_epochs=num_epochs,
                                         learning_rate=learning_rate,
                                         learning_rate_decay=learning_rate_decay,
                                         momentum=momentum,
                                         nesterov=nesterov,
                                         verbose=verbose)
    else:
        mnist_digit_classifier = None

    # Create a Qt application
    app = QApplication(sys.argv)

    # Create a window
    window = MnistClassifierDemonstrator(mnist_digit_classifier=mnist_digit_classifier)

    # Show window
    window.show()

    # Enter Qt application main loop
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
