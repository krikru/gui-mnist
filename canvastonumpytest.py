# Import PySide classes
import sys
from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np

# Image options
w = h = 28
pen_width = 2.0

# Classification task
num_classes = 10

# Layout options
spacing = 40
canvas_scale = 10
button_height = spacing

# Screen options
inv_screen_gamma = 1 / 2.2


class Canvas(QWidget):
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
        self.small_image.fill(qRgb(255, 255, 255))
        self.large_image = QImage(self.scaled_w, self.scaled_h, QImage.Format_RGB32)
        self.large_image.fill(qRgb(255, 255, 255))

        # Create pen
        self.pen = QPen()
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setWidthF(scale * pen_width)

        # There is currently no path
        self.currentPath = None

    def _get_painter(self, paintee):
        painter = QPainter(paintee)
        painter.setPen(self.pen)
        painter.setRenderHint(QPainter.Antialiasing, True)
        #painter.scale(self.scale, self.scale)
        return painter

    def getContent(self):
        return np.asarray(self.small_image.constBits()).reshape((self.h, self.w, -1))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.currentPath = QPainterPath()
            self.currentPath.moveTo(event.pos())
            painter = self._get_painter(self.large_image)
            painter.drawPoint(event.pos())
            painter.end()
            self.repaint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.currentPath is not None:
            self.currentPath.lineTo(event.pos())
            self.repaint()

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton) and self.currentPath is not None:
            # Finish line and draw path
            self.currentPath.lineTo(event.pos())
            painter = self._get_painter(self.large_image)
            painter.drawPath(self.currentPath)
            painter.end()
            self.currentPath = None
        elif event.button() == Qt.RightButton:
            self.large_image.fill(qRgb(255, 255, 255))

        # Downsample image
        self.small_image = self.large_image.scaled(self.w, self.h, mode=Qt.SmoothTransformation)
        self.repaint()

    def paintEvent(self, event):
        paint_rect = event.rect()  # Only paint the surface that needs painting
        painter = self._get_painter(self)

        # Draw image
        painter.scale(self.scale, self.scale)
        painter.drawImage(paint_rect, get_gamma_corrected_qimage(self.small_image), paint_rect)

        painter.end()

        painter = self._get_painter(self)

        if self.currentPath is not None:
            painter.drawPath(self.currentPath)


class BarChartBar(QWidget):
    def __init__(self, parent, value=None, max_value=None, foreground_color=Qt.GlobalColor.red,
                 background_color=Qt.GlobalColor.white, *argv, **kwargs):
        super().__init__(parent, *argv, **kwargs)
        self.set_value(value, max_value)
        self.foreground_color = QColor(foreground_color)
        self.background_color = QColor(background_color)

    def set_value(self, value, max_value):
        self.value     = value
        self.max_value = max_value

    def paintEvent(self, event):
        if self.value is None or self.max_value is None:
            return
        painter = QPainter(self)
        f_width = (self.value/self.max_value) * self.width()
        f_width_whole_part = min(int(f_width), self.width()-1)
        f_width_fractional_part = f_width - f_width_whole_part
        b_width_whole_part = self.width() - f_width_whole_part - 1
        middle_color = interpolate_qcolor(self.foreground_color, self.background_color, f_width_fractional_part)
        gamma_corrected_middle_color = get_gamma_corrected_qcolor(middle_color)
        painter.fillRect(QRect(0, 0, f_width_whole_part, self.height()), self.foreground_color)
        painter.fillRect(QRect(f_width_whole_part, 0, 1, self.height()), gamma_corrected_middle_color)
        painter.fillRect(QRect(f_width_whole_part+1, 0, b_width_whole_part, self.height()), self.background_color)


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
        self.drawing_widgets = []
        for idx in range(self.num_bars):
            drawing_widget = BarChartBar(self)
            self.drawing_widgets.append(drawing_widget)
            layout.addRow(str(idx), drawing_widget)

        self.update_bars()

    def set_values(self, values, max_value=None):
        self.values = values
        self.max_value = max_value
        self.update_bars()

    def update_bars(self):
        max_value = self.max_value if self.max_value else max([val for val in self.values if val is not None] + [0])
        for idx in range(self.num_bars):
            self.drawing_widgets[idx].set_value(self.values[idx], max_value)


class MnistClassifierDemonstrator(QMainWindow):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('Canvas to Numpy')

        # Create a canvas
        self.canvas = Canvas(self, w, h, pen_width, canvas_scale)
        layout.addWidget(self.canvas)

        # Create button for getting content
        self.button = QPushButton(self)
        self.button.setText("Get canvas content")
        self.button.clicked.connect(lambda: button_clicked(self.canvas))
        layout.addWidget(self.button)

        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.bar_chart = BarChart(self, num_classes)
        self.bar_chart.set_values(values=[1/3]*num_classes, max_value=1)
        layout.addWidget(self.bar_chart)


def get_gamma_corrected_qcolor(qcolor):
    c = np.append(np.array([qcolor.redF(), qcolor.greenF(), qcolor.blueF()]) ** inv_screen_gamma, qcolor.alphaF())
    return QColor(*((255 * c).astype(int).tolist()))


def interpolate_qcolor(front, back, alpha):
    [c1, c2] = [np.append(qcolor.alphaF() * np.array([qcolor.redF(), qcolor.greenF(), qcolor.blueF()]), qcolor.alphaF())
                for qcolor in [front, back]]
    c = alpha * c1 + (1 - alpha) * c2
    c[:3] *= 1 / c[3] if c[3] > 0 else 0
    return QColor(*((255 * c).astype(int).tolist()))


def get_gamma_corrected_qimage(qimage):
    corrected = qimage.copy()
    for x in range(corrected.width()):
        for y in range(corrected.height()):
            curr_rgba = corrected.pixel(x, y)
            new_rgb = sum([int(255 * (((curr_rgba >> i*8) & 255) / 255) ** inv_screen_gamma) << i*8 for i in range(3)])
            new_rgba = new_rgb + (curr_rgba & (255 << 24))
            corrected.setPixel(x, y, new_rgba)
    return corrected


def button_clicked(canvas):
    gray_scale_canvas_content = canvas.getContent()[:, :, 0]
    for row in gray_scale_canvas_content:
        for element in row:
            print(' ' if element < 128 else 'X', end='')
        print()
    print(flush=True)


def main():

    # Create a Qt application
    app = QApplication(sys.argv)

    # Create a window
    window = MnistClassifierDemonstrator()

    # Show window
    window.show()

    # Enter Qt application main loop
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
