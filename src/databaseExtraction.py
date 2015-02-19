#!/usr/bin/python
"""Run this program to read the pictures from a databases,
	extract the faces from thoses pictures and save them
		Args:
		pathSrc: Path to a folder with subfolders representing the subjects (persons). 
		pathDst: Path to a folder were the extract faces will be saved

"""
import numpy as np
import cv2
import cv
import uuid
import functions
import classes
import sys
import os


if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "USAGE: databaseExtraction.py </path/to/src/images> </path/to/dst/images>"
		print "OR : to put in the /tmp filder by default"
		print "USAGE: databaseExtraction.py </path/to/src/images>"
		sys.exit()
	

	WindowsName="img"
	faceDetected=False

	faceDetect = classes.FaceDetection()
	imgCrop = classes.ImageCropper("Stef")

	[X,y] = functions.read_images(sys.argv[1])

	for i in xrange(len(X)):
		print "i is"
		print i
		currentImage=X[i]
		folderNumber = str(y[i])
		# Uncomment this to create a folder for each user (but there might be rights problem)
		#if not os.path.exists(sys.argv[2]+"/person"+folderNumber): 
		#	os.makedirs("/person"+folderNumber)
		faces = faceDetect.detect(currentImage)
		cv2.imshow(WindowsName,currentImage)
		if len(faces) >= 1:
			for j in xrange(len(faces)):
				face = faceDetect.faceSelector(faces,j)
				if len(sys.argv) >= 2:
					imgCrop.crop(currentImage, face, sys.argv[2]+"person"+folderNumber)
				else:
					imgCrop.crop(currentImage, face)	
			cv2.imshow(WindowsName,currentImage)
cv2.destroyAllWindows()