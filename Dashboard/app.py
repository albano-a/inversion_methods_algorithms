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
from OpenGL.GL import *
import numpy as np
from scipy import signal
import sys

plt.style.use(['bmh'])

class Ricker:
    def __init__(self, peak_freq, samples, dt, canvas):
        self.peak_freq = peak_freq
        self.samples = samples
        self.dt = dt
        self.canvas = canvas

    def wavelet(self):
        twlet = np.arange(self.samples) * (self.dt / 1000)
        twlet = np.concatenate((np.flipud(-twlet[1:]), twlet), axis=0)
        wlet = (1. -2.*(np.pi**2)*(self.peak_freq**2)*(twlet**2))*np.exp(-(np.pi**2)*(self.peak_freq**2)*(twlet**2))
        return twlet, wlet

    def plot(self):
        twlet, wlet = self.wavelet()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(twlet, wlet)
        ax.set_title('Ricker Wavelet')
        self.canvas.figure.set_tight_layout(True)
        self.canvas.draw()

class Butterworth:
    def __init__(self, peak_freq, low_freq,samples, dt, canvas):
        self.peak_freq = peak_freq
        self.low_freq = low_freq
        self.samples = samples
        self.dt = dt
        self.canvas = canvas
        
    def wavelet(self):
        twlet = np.arange(self.samples) * (self.dt / 1000)
        twlet = np.concatenate((np.flipud(-twlet[1:]), twlet), axis=0)
        # Create impulse signal
        imp = signal.unit_impulse(twlet.shape[0], 'mid')

        # Apply high-pass Butterworth filter
        fs = 1000 * (1 / self.dt)
        b, a = signal.butter(4, self.peak_freq, fs=fs)
        response_zp = signal.filtfilt(b, a, imp)

        # Apply low-pass Butterworth filter
        low_b, low_a = signal.butter(2, self.low_freq, 'hp', fs=fs)
        butter_wvlt = signal.filtfilt(low_b, low_a, response_zp)

        return twlet, butter_wvlt
    
    def plot(self):
        twlet, butter_wvlt = self.wavelet()
        ax = self.canvas.figure.add_subplot(121)
        ax.plot(twlet, butter_wvlt)
        ax.set_title('Butterworth Wavelet')
        ax1 = self.canvas.figure.add_subplot(122)
        ax1.plot(twlet, butter_wvlt)
        ax1.set_title('Butterworth Wavelet2')
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
        self.peak_freq = self.peakFreqInput.text()
        self.low_freq = self.lowFreqInput.text()
        self.samples = self.sampleInput.text()
        self.dt = self.dtInput.text()
        self.wavelet = self.waveletsComboBox.currentText()
        
        if self.wavelet == 'Ricker':
            self.ricker = Ricker(float(self.peak_freq), 
                                 int(self.samples), 
                                 float(self.dt), 
                                 self.canvas)
            return self.ricker.plot()
        if self.wavelet == 'ButterWorth':
            self.butterworth = Butterworth(float(self.peak_freq), 
                                           float(self.low_freq), 
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