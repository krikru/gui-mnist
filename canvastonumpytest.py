# Import PySide classes
import sys
from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np

# Image options
w = h = 28
pen_width = 2.0

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


class MnistClassifierDemonstrator(QMainWindow):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        # Geometry
        sw = canvas_scale * w
        sh = canvas_scale * h

        self.resize(2 * spacing + sw, 3 * spacing + sh + button_height)
        self.setWindowTitle('Canvas to Numpy')

        # Create a canvas
        self.canvas = Canvas(self, w, h, pen_width, canvas_scale)
        self.canvas.setGeometry(QRect(spacing, spacing, sw, sh))

        # Create button for getting content
        self.button = QPushButton(self)
        self.button.setText("Get canvas content")
        self.button.setGeometry(QRect(spacing, 2 * spacing + sh, sw, button_height))
        self.button.clicked.connect(lambda: button_clicked(self.canvas))


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
