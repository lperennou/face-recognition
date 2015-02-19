import cv2
import functions
import numpy as np

class ModelPCAGlobal:
	def __init__(self, numberOfEigenfaces=None, threshold=None, train_path=None, img_size=None, training = None):
		self.trainImagesAndLabels = training

		if (train_path is not None):
			self.train_path = train_path
		else: 
			self.train_path = "data/training/"  
		if (img_size is not None):
			self.img_size = img_size
		else: 
			self.img_size = (100,100)    

		if (numberOfEigenfaces is not None):
			self.numberOfEigenfaces = numberOfEigenfaces
		else:
			self.numberOfEigenfaces = 9
		if (threshold is not None):
			self.threshold = threshold
		else:
			self.threshold = 4500
		print "model Global numberOfEigenfaces:"
		print self.numberOfEigenfaces
		print "model Global threshold"
		print self.threshold

		self.model = cv2.createEigenFaceRecognizer(self.numberOfEigenfaces, self.threshold)
		self.train()

	def train(self):
		if self.trainImagesAndLabels is None :
			self.trainImagesAndLabels = functions.read_images(self.train_path,self.img_size)
		self.model.train(self.trainImagesAndLabels[0], self.trainImagesAndLabels[1])

	def predict(self, image):
		testing = cv2.resize(image, self.img_size)  #We resize the image at the model format images size
		#cv2.imshow("img for recognize",testing)
		[label, confidence] = self.model.predict(testing)
		return [int(label), int(confidence)]
	
	def evaluate(self, path_to_test_images_known = None, path_to_test_images_unknown = None, test_images_known = None, test_images_unknown = None):
		false_positive = 0 #Number of unknown persons who were classified as known
		false_negative = 0 #Number of known persons who were classified as unknown
		misclassified_persons = 0 #Number of known persons who were classified as another known person
		
		#First, test the set of known persons and count the false negatives and misclassifications
		if test_images_known is None : 
			testImagesAndLabels = functions.read_images(path_to_test_images_known,self.img_size)
		else :
			testImagesAndLabels = test_images_known
		known_images = len(testImagesAndLabels[1])
		for i in range(known_images): #for all test images
			prediction = self.predict(testImagesAndLabels[0][i])

			if prediction[0] == -1 : # if the person is detected as unknown
				false_negative = false_negative + 1
			elif prediction[0] != testImagesAndLabels[1][i] : #if the predicted label is misclassified
				misclassified_persons = misclassified_persons + 1

		#Second, test the set of unknown persons and count the false positive
		if test_images_unknown is None : 
			testImagesAndLabels = functions.read_images(path_to_test_images_unknown,self.img_size)
		else :
			testImagesAndLabels = test_images_unknown
		unknown_images = len(testImagesAndLabels[1])
		for i in range(unknown_images): #for all test images
			prediction = self.predict(testImagesAndLabels[0][i])
			if prediction[0] != -1 : # if the person is detected as known
				false_positive = false_positive + 1

		return [float(false_positive)/unknown_images, float(false_negative)/(known_images), float(misclassified_persons)/(known_images-false_negative), float(false_positive+false_negative+misclassified_persons)/(known_images+unknown_images) ]


class ModelPerPerson:
	def __init__(self, numberOfEigenfaces=None,threshold=None, train_path=None, img_size=None, training = None):
		self.trainImagesAndLabels = training

		if (train_path is not None):
			self.train_path = train_path
		else: 
			self.train_path = "data/training/"  
		if (img_size is not None):
			self.img_size = img_size
		else: 
			self.img_size = (100,100)    

		if (numberOfEigenfaces is not None):
			self.numberOfEigenfaces = numberOfEigenfaces
		else:
			self.numberOfEigenfaces = 9
		if (threshold is not None):
			self.threshold = threshold
		else:
			self.threshold = 2000
		print "model Per person numberOfEigenfaces:"
		print self.numberOfEigenfaces
		print "model Per person threshold"
		print self.threshold
		# Here we create all the models 
		# Read the images that has been extract 
		if(training is None) :
			training = functions.read_images(self.train_path,self.img_size) # X[0] contains the image 0 , and y[0] the corresponding label
		# Return a table containing on each row one model data for a person 
		self.modelsData = functions.divideModelsByPerson(training[0], training[1])   # modelsData[0][0] contains a list of all the images, modesData[0][1] contains a list of 0.
															#   We need to keep this structure to them build models, because this is the way it should be
															#  given to the openCV function.
															# So models[0] contains everything we need to give to the openCV function to make it build 
															#the model for the person 0""" 
		self.models = [] # models[0] contains the model of the person 0, models[1] the model of the person 1...
		
		for i in xrange(len(self.modelsData)):
			self.models.append(cv2.createEigenFaceRecognizer(self.numberOfEigenfaces))
		# We train a first time each model	
		self.train()

	def train(self):
		for i in xrange(len(self.models)):
			self.models[i].train(np.asarray(self.modelsData[i][0]),np.asarray(self.modelsData[i][1]))

	def predict(self, image): 
		testing = cv2.resize(image, self.img_size)  #We resize the image at the model format images size
		#We apply the prediction on each model
		testResult = []
		for i in xrange(len(self.models)):
			testResult.append(self.models[i].predict(testing))
		testResult = np.asarray(testResult, np.int32)
		bestResult = testResult[np.argmin(testResult[:,1])]	
		if bestResult[1] >= self.threshold:
			bestResult[0] = -1
		return bestResult
	
	def evaluate(self, path_to_test_images_known = None, path_to_test_images_unknown = None, test_images_known = None, test_images_unknown = None):
		false_positive = 0 #Number of unknown persons who were classified as known
		false_negative = 0 #Number of known persons who were classified as unknown
		misclassified_persons = 0 #Number of known persons who were classified as another known person
		
		#First, test the set of known persons and count the false negatives and misclassifications
		if test_images_known is None : 
			testImagesAndLabels = functions.read_images(path_to_test_images_known,self.img_size)
		else :
			testImagesAndLabels = test_images_known
		known_images = len(testImagesAndLabels[1])
		for i in range(known_images): #for all test images
			prediction = self.predict(testImagesAndLabels[0][i])
			if prediction[0] == -1 : # if the person is detected as unknown
				false_negative = false_negative + 1
			elif prediction[0] != testImagesAndLabels[1][i] : #if the predicted label is misclassified
				misclassified_persons = misclassified_persons + 1

		#Second, test the set of unknown persons and count the false positive
		if test_images_unknown is None : 
			testImagesAndLabels = functions.read_images(path_to_test_images_unknown,self.img_size)
		else :
			testImagesAndLabels = test_images_unknown
		unknown_images = len(testImagesAndLabels[1])
		for i in range(unknown_images): #for all test images
			prediction = self.predict(testImagesAndLabels[0][i])
			if prediction[0] != -1 : # if the person is detected as known
				false_positive = false_positive + 1

		return [float(false_positive)/unknown_images, float(false_negative)/known_images, float(misclassified_persons)/(known_images-false_negative), float(false_positive+false_negative+misclassified_persons)/(known_images+unknown_images) ]

