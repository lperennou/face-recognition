#Classes Utils
import os
import sys
import cv2
import numpy as np
import cv
import uuid
import PyQt4 as qt
from PyQt4 import QtGui
import sys
import classes
import functions
import models

cameraWidth=0
cameraHeight=0

class Face:
	def __init__(self,faceArray):
		self.x,self.y,self.w,self.h=faceArray[0],faceArray[1],faceArray[2],faceArray[3]
	
class Camera:
	def __init__(self,CameraID):
		self.camera = cv2.VideoCapture(CameraID)
		global cameraWidth
		global cameraHeight
		cameraWidth = self.camera.get(cv.CV_CAP_PROP_FRAME_WIDTH)
		cameraHeight = self.camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

	def read(self):
		ret, img = self.camera.read()
		return img

	def release(self):
		self.camera.release()
	def convertFrame(self,imgB):
		img=cv2.cvtColor(imgB,cv2.COLOR_BGR2RGB)
		height,width=img.shape[:2]
		img=QtGui.QImage(img,width,height,QtGui.QImage.Format_RGB888)
		QTimg=QtGui.QPixmap.fromImage(img)
		return QTimg

class FaceDetection:
	def toGrey(self, coloredImage):
		return cv2.cvtColor(coloredImage, cv2.COLOR_BGR2GRAY)
	def detect(self, greyimage):
		return self.face_cascade.detectMultiScale(greyimage, 1.3, 5)
	def drawFace(self,face,coloredImage):
	  	cv2.rectangle(coloredImage,(face.x,face.y),(face.x+face.w,face.y+face.h),(255,0,0),2)
	  	cv2.rectangle(coloredImage,(face.x-int(0.5*face.w),face.y-int(0.5*face.h)),(face.x+int(1.5*face.w),face.y+int(1.5*face.h)),(0,255,0),2)
		return coloredImage 
	def faceSelector(self,faces, num=None):
		if ( num is not None ):
			face=Face(faces[num])
		else:
			face = Face(faces[0])
		return face

	def setCamera(self,cam):
		self.camera=cam

	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
		

class FaceTraking:
	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
	def detect(self, greyimage,roi):
		a,b,c,d=roi.y-int(0.5*roi.h),roi.y+int(1.5*roi.h),roi.x-int(0.5*roi.w),roi.x+int(1.5*roi.w)
		if a < 0: a=0
		if b >  cameraHeight: b = cameraHeight-1
		if c < 0: c= 0
		if d > cameraWidth: d = cameraWidth-1
		roi_image = greyimage[a:b,c :d]
		faces = self.face_cascade.detectMultiScale(roi_image, 1.3, 5)	#return a ROI of a smaller image
		cp = []
		for (x,y,w,h) in faces:#Correct the ROI
			f = [0,0,0,0]						
			f[0],f[1],f[2],f[3] = x+c,y+a,w,h
			cp.append(f)
		return cp
		

class ImageCropperForEnroll:
	def __init__(self,destination_path):
		self.destination=destination_path

	def crop(self,image, face):
		sub_face = image[face.y:face.y+face.h, face.x:face.x+face.w]
		cv2.equalizeHist(sub_face, sub_face)
		face_file_name = self.destination+"/face_" + str(uuid.uuid4()) + ".jpg"
		cv2.imwrite(face_file_name, sub_face)

class ImageCropperForRecognize:
	def crop(self,image, face):
		sub_face = image[face.y:face.y+face.h, face.x:face.x+face.w]
		cv2.equalizeHist(sub_face, sub_face)
		return np.asarray(sub_face)


class Program:
 	def __init__(self,imageBox,globalPcaBox, personalPcaBox):
 		self.imageBox=imageBox
 		self.globalPcaBox=globalPcaBox
 		self.personalPcaBox=personalPcaBox
 		self.cropping=25
 		self.numberOfEigenfaces = 15
 		self.threshold = 3000
 		self.modelPCAGlobal = models.ModelPCAGlobal()
 		self.modelPCAPerPerson = models.ModelPerPerson()
 		self.faceDetectRunning = False
 		self.testingImage = []
 		self.cam = None


	def startFaceDetect(self,status=None):
		self.windowsName="0"
		cv2.namedWindow(self.windowsName, cv2.WINDOW_NORMAL)
		self.faceDetectRunning = True
		faceDetected=False
		if self.cam == None : # create camera only if 'start detect' button has not been pressed
			self.cam=classes.Camera(0)
		faceDetect = classes.FaceDetection()
		faceDetect.setCamera(self.cam)
		faceTrack = classes.FaceTraking()
		cropping = 0

		while(self.faceDetectRunning):
			print 'loop\n'
			cv2.waitKey(5)
			coloredImage = self.cam.read()
			if coloredImage is None:
				print "coloredImage is None "
				break #exit the loop because the cam was turn off
			greyimage = faceDetect.toGrey(coloredImage)
			if faceDetected==False:
				faces = faceDetect.detect(greyimage)
			else:
				#faces = faceDetect.detect(greyimage)
				faces = faceTrack.detect(greyimage, face)
			if len(faces) >= 1:
				faceDetected=True
				face = faceDetect.faceSelector(faces)
				coloredImage=faceDetect.drawFace(face,coloredImage)
				if status == 'enroll' : #if we want to enroll a user
					cropping=cropping+1
					self.cropper.crop(greyimage, face)
					if cropping == self.cropping : #exit the loop after all picture have been taken
						print "cropping is done"
						self.modelPCAGlobal.train()
						self.ModelPerPerson.train() # We got a new person, we therefore need to train again
						break
				elif status == 'recognize' : #if we want to recognize a user
					self.testingImage = self.cropper.crop(greyimage,face)
					cv2.equalizeHist(self.testingImage, self.testingImage)
					self.imageBox.setPixmap(self.cam.convertFrame(coloredImage))
					break #exit the loop after the picture is saved
			else:
				faceDetected=False
			qTimage=self.cam.convertFrame(coloredImage)
			self.imageBox.setPixmap(qTimage)
		
		#Destroy all ressources (webcam, windows, others?)
		cv2.destroyWindow("webcam")
		#self.cam.release()

	def enroll(self, username):
		userpath=os.path.abspath("data/training/"+str(username)+"_"+str(functions.getNumberOfPersonInFolder("data/training")))
		self.cropper = classes.ImageCropperForEnroll(userpath)
		if not os.path.exists(userpath):
			os.makedirs(userpath)
		self.startFaceDetect('enroll')
		self.stop()
		

	def recognize(self):
		self.cropper = classes.ImageCropperForRecognize()
		self.startFaceDetect('recognize')
		resultPCAGlobal = self.modelPCAGlobal.predict(self.testingImage)
		resultPCAPerPerson = self.modelPCAPerPerson.predict(self.testingImage)

		#Integration of the mesure of confidence by tristan
		print "resultPCAGlobal[0]"
		print resultPCAGlobal[0]

		print "resultPCAPerPerson[0]"
		print resultPCAPerPerson[0]

		resultPCAPerPerson[0]
		print "np.asarray(self.modelPCAPerPerson.modelsData)"
		print np.asarray(self.modelPCAPerPerson.modelsData[1][0]).shape
		print "self.testingImage shape"
		print np.asarray(self.testingImage).shape
		test_resized=cv2.resize(self.testingImage,(100,100))

		print "result Global PCA"
		print resultPCAGlobal
		print "resulr PCA per Person"
		print resultPCAPerPerson
		if (resultPCAGlobal[0]==-1):
			confidencePCAGlobal=-1
			confidenceProjected=-1
		"""SEUIL else:
			#confidencePCAGlobal = functions.probaCorrectRecognition(resultPCAGlobal[0], self.modelPCAPerPerson.modelsData, self.modelPCAPerPerson.models, test_resized, (100,100))
			confidenceProjected=functions.probaCorrectRecognition_Projected(resultPCAGlobal[0], self.modelPCAPerPerson.modelsData, self.modelPCAPerPerson.models, test_resized, (100,100),10)

		
		if (resultPCAPerPerson[0]==-1):
			confidenceOnePCAPerPerson=-1
			confidenceProjectedPerpers=-1
		else:
			#confidenceOnePCAPerPerson= functions.probaCorrectRecognition(resultPCAPerPerson[0], self.modelPCAPerPerson.modelsData, self.modelPCAPerPerson.models, test_resized, (100,100))
			confidenceProjectedPerpers=functions.probaCorrectRecognition_Projected(resultPCAPerPerson[0], self.modelPCAPerPerson.modelsData, self.modelPCAPerPerson.models, test_resized, (100,100),10)






		print "------ Projected --------"
		print "confidence global"
		print confidenceProjected
		print "confidence per person"
		print confidenceProjectedPerpers      SEUIL"""


#		print "------ Canonic --------"
#		print "confidencePCAGlobal"
#		print confidencePCAGlobal
#		print "confidencePCAPerPerson"
#		print confidenceOnePCAPerPerson




		imageResultGlobalPCA = functions.retreiveAnImageFromFolder("data/training/",resultPCAGlobal[0])
		imageResultPCAPerPerson = functions.retreiveAnImageFromFolder("data/training/",resultPCAPerPerson[0])
		self.globalPcaBox.setPixmap(imageResultGlobalPCA)
		self.personalPcaBox.setPixmap(imageResultPCAPerPerson)
		self.stop()
		
			
	
	def stop(self):
		self.faceDetectRunning = False
		#wait(1)

		# cv2.destroyAllWindows()

	def quit(self):
		cv2.destroyAllWindows()
		qt.QtCore.QCoreApplication.quit()
		cv2.destroyAllWindows()
		exit()

class QTWindow(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')
        self.imageBox = QtGui.QLabel(self)
        self.globalPcaBox = QtGui.QLabel(self)
       	self.personalPcaBox = QtGui.QLabel(self)
        self.program = Program(self.imageBox,self.globalPcaBox,self.personalPcaBox)
        self.start_reco_button = QtGui.QPushButton('Start Detect',self)
        self.start_reco_button.clicked.connect(self.program.startFaceDetect)
        self.enroll_button = QtGui.QPushButton('Enroll me',self)
        self.enroll_button.clicked.connect(self.enroll)
        self.recognize_button = QtGui.QPushButton('recognize me',self)
        self.recognize_button.clicked.connect(self.program.recognize)
        self.stop_button = QtGui.QPushButton('Stop',self)
        self.stop_button.clicked.connect(self.program.stop)
        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(self.program.quit)

        vboxCont = QtGui.QLabel(self)
        vbox = QtGui.QVBoxLayout(vboxCont)
        vbox.addWidget(self.start_reco_button)
        vbox.addWidget(self.enroll_button)
        vbox.addWidget(self.recognize_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.quit_button)
        vboxCont.setGeometry(0,0,150,150)

        self.imageBox.setScaledContents(True)
        self.imageBox.setGeometry(0,0,1280,720)
        self.globalPcaBox.setGeometry(0,150,300,300)
        self.personalPcaBox.setGeometry(0,450,300,300)
        self.setGeometry(0,0,1280,720)
        self.show()

    def enroll(self):
    	if not self.program.faceDetectRunning :
    		return
        (text,truth)=QtGui.QInputDialog.getText(self,"Get text","User name",QtGui.QLineEdit.Normal)
        if truth:
            #The user has accepted the edit, he/she has clicked OK
            self.program.enroll(text)
        else:
            #The user has not accepted the edit, he/she has clicked Cancel
            print "No change"

  



