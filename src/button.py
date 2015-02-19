
import PyQt4 as qt
from PyQt4 import QtGui
import sys
import cv2
class Capture():
    def __init__(self):
        self.capturing = False
        self.c = cv2.VideoCapture(0)
    def startCapture(self):
        print "pressed start"
        self.capturing = True
        cap = self.c
        while(self.capturing):
            ret, frame = cap.read()
            cv2.imshow("Capture", frame)
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def endCapture(self):
        print "pressed End"
        self.capturing = False
        # cv2.destroyAllWindows()

    def quitCapture(self):
        print "pressed Quit"
        cap = self.c
        cv2.destroyAllWindows()
        cap.release()
        QtCore.QCoreApplication.quit()
class Window(QtGui.QWidget):
    def __init__(self):

        c = cv2.VideoCapture(0)

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.start_button = QtGui.QPushButton('Start',self)
        self.start_button.clicked.connect(lambda : startCapture(c, True))

        self.end_button = QtGui.QPushButton('End',self)
        self.end_button.clicked.connect(lambda : endCapture(c))

        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(lambda : quit(c))

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        self.setLayout(vbox)
        self.setGeometry(100,100,200,200)
        self.show()
        self.capture = Capture()
     	self.start_button = QtGui.QPushButton('Start',self)
     	self.start_button.clicked.connect(self.capture.startCapture)
     	self.end_button = QtGui.QPushButton('End',self)
     	self.end_button.clicked.connect(self.capture.endCapture)
     	self.quit_button = QtGui.QPushButton('Quit',self)
     	self.quit_button.clicked.connect(self.capture.quitCapture)

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
