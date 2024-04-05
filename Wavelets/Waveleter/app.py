from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QUrl, QResource, QCoreApplication, QDir
from PyQt6.QtGui import (QAction, QKeySequence, QDesktopServices,
                        QFileSystemModel, QStandardItemModel, QStandardItem,
                        QFontMetricsF, QTextOption, QFont, QKeyEvent, QPainter)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidgetItem,QGraphicsScene,
                            QFileDialog, QMessageBox, QColorDialog, QFontDialog,
                            QInputDialog, QTreeWidgetItem, QVBoxLayout, QWidget, QSizePolicy)
import PyQt6.Qsci as Qsci
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from main import Ui_MainWindow
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy import signal
import sys

plt.style.use(['bmh'])

class Ricker:
    def __init__(self, high_freq, samples, dt, canvas):
        self.high_freq = high_freq
        self.samples = samples
        self.dt = dt
        self.canvas = canvas

    def wavelet(self):
        twlet = np.arange(self.samples) * (self.dt / 1000)
        twlet = np.concatenate((np.flipud(-twlet[1:]), twlet), axis=0)
        wlet = (1. -2.*(np.pi**2)*(self.high_freq**2)*(twlet**2))*np.exp(-(np.pi**2)*(self.high_freq**2)*(twlet**2))
        return twlet, wlet

    def plot(self):
        twlet, wlet = self.wavelet()

        fft_r = abs(np.fft.rfft(wlet))
        freqs_r = np.fft.rfftfreq(twlet.shape[0], d=4/1000)
        fft_r = fft_r / np.max(fft_r)

        ax = self.canvas.figure.add_subplot(211)
        ax.plot(twlet, wlet)
        ax.set_title('Ricker Wavelet')
        ax1 = self.canvas.figure.add_subplot(212)
        ax1.plot(freqs_r, fft_r)
        ax1.set_title('Ricker Spectrum')
        self.canvas.figure.set_tight_layout(True)
        self.canvas.draw()

class Butterworth:
    def __init__(self, high_freq, low_freq,samples, dt, canvas):
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.samples = samples
        self.dt = dt
        self.canvas = canvas       # wavelet


    def wavelet(self):
        twlet = np.arange(self.samples) * (self.dt / 1000)
        twlet = np.concatenate((np.flipud(-twlet[1:]), twlet), axis=0)
        # Create impulse signal
        imp = signal.unit_impulse(twlet.shape[0], 'mid')

        # Apply high-pass Butterworth filter
        fs = 1000 * (1 / self.dt)
        b, a = signal.butter(4, self.high_freq, fs=fs)
        response_zp = signal.filtfilt(b, a, imp)

        # Apply low-pass Butterworth filter
        low_b, low_a = signal.butter(2, self.low_freq, 'hp', fs=fs)
        butter_wvlt = signal.filtfilt(low_b, low_a, response_zp)

        return twlet, butter_wvlt

    def plot(self):
        twlet, butter_wvlt = self.wavelet()

        fft_b = abs(np.fft.rfft(butter_wvlt))
        freqs_b = np.fft.rfftfreq(twlet.shape[0], d=4/1000)
        fft_b = fft_b / np.max(fft_b)

        ax = self.canvas.figure.add_subplot(211)
        ax.plot(twlet, butter_wvlt)
        ax.set_title('Butterworth Wavelet')
        ax1 = self.canvas.figure.add_subplot(212)
        ax1.plot(freqs_b, fft_b)
        ax1.set_title('Butterworth Spectrum')
        self.canvas.figure.set_tight_layout(True)
        self.canvas.draw()

class MyGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        self.setupUi(self)  # call the setupUi method of the Ui_MainWindow class

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # create a QGraphicsScene
        self.scene = QGraphicsScene()

        # add the FigureCanvas to the QGraphicsScene
        self.scene.addWidget(self.canvas)

        # set the QGraphicsScene for the QGraphicsView
        self.graphicsView.setScene(self.scene)

        self.waveletsComboBox.addItems(['Ricker', 'ButterWorth'])

        self.waveletPlotBtn.clicked.connect(self.plot)
        self.exportFigPlot.clicked.connect(self.export_figure)

    def plot(self):
        self.figure.clear()
        self.high_freq = self.peakFreqInput.text()
        self.low_freq = self.lowFreqInput.text()
        self.samples = self.sampleInput.text()
        self.dt = self.dtInput.text()
        self.wavelet = self.waveletsComboBox.currentText()

        if self.wavelet == 'Ricker':
            self.ricker = Ricker(float(self.high_freq),
                                 int(self.samples),
                                 float(self.dt),
                                 self.canvas)
            return self.ricker.plot()
        if self.wavelet == 'ButterWorth':
            if not self.high_freq or not self.low_freq:
                QMessageBox.warning(self, 'Aviso', 'Frequência Alta e Frequência Baixa são obrigatórios')
            else:
                try:
                    high_freq = float(self.high_freq)
                    low_freq = float(self.low_freq)
                except ValueError:
                    QMessageBox.warning(self, 'Aviso', 'Frequência Alta e Frequência Baixa devem ser números válidos')
                    return

                self.butterworth = Butterworth(high_freq,
                                                low_freq,
                                                int(self.samples),
                                                float(self.dt),
                                                self.canvas)
                return self.butterworth.plot()

    def export_figure(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  'Save File',
                                                  '',
                                                  'Images (*.png *.jpg *.bmp *.svg)')
        if filename:
            self.canvas.figure.savefig(filename)


def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()