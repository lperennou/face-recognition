#!/usr/bin/python
import numpy as np
import cv2
import cv
import uuid
from PyQt4 import QtGui
import sys
import classes

if __name__ == "__main__":

	app = QtGui.QApplication(sys.argv)
	window=classes.QTWindow()
	sys.exit(app.exec_())
	cv2.destroyAllWindows()