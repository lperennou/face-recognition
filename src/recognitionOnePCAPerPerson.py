#!/usr/bin/python
import numpy as np
import cv2
import cv
import uuid
import functions
import os
import sys

sz=(64,64)
numberOfEigenfaces = 20
threshold = 2000

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: recognitionOnePCAPerPerson.py </path/to/training/images> </path/to/testing/image.jpg>"
		sys.exit()

	# Read the images that has been extract 
	[X,y] = functions.read_images(sys.argv[1],sz) # X[0] contains the image 0 , and y[0] the corresponding label


	# Convert labels to 32bit integers. This is a workaround for 64bit machines
	y = np.asarray(y, dtype=np.int32)

	# Return a table containing on each row one model data for a person 
	modelsData = functions.divideModelsByPerson(X,y)   # modelsData[0][0] contains a list of all the images, modesData[0][1] contains a list of 0.
														#   We need to keep this structure to them build models, because this is the way it should be
														#  given to the openCV function.
														# So models[0] contains everything we need to give to the openCV function to make it build 
														#the model for the person 0""" 
	#Create a window
	WindowsName="img"	
	cv2.namedWindow(WindowsName, cv2.WINDOW_NORMAL)	


	#Do a PCA for each model
	models = [] # models[0] contains the model of the person 0, models[1] the model of the person 1...
	for i in xrange(len(modelsData)):
		models.append(cv2.createEigenFaceRecognizer(numberOfEigenfaces, threshold))
		models[i].train(np.asarray(modelsData[i][0]), np.asarray(modelsData[i][1]))
	#Test with the given image 
	testResult = []
	testing = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
	if (sz is not None ):
		testing = cv2.resize(testing, sz)
	else:
		testing = cv2.resize(testing, (200,200))
	cv2.imshow("img",testing)




	#We apply the prediction on each model
	for i in xrange(len(models)):
		testResult.append(models[i].predict(testing))


	#We look for on the result that are no negative and find the best confidence between them
	positiveTestResult = []
	for i in xrange(len(testResult)):
		if testResult[i][0] != (-1):
			positiveTestResult.append(testResult[i])	


	#Partie Tristan
	#Liste des variables
	# 
	sample_per_ID=5
	person_ID=4
	temp_img = []
	for i in xrange(len(modelsData)):
		label=modelsData[i][1][0]   						# label des personnes de la base
		if label==person_ID:  						# S'il s'agit de la personne reconnue
			listImagesPerson=modelsData[i][0]						# liste d'image pour une meme personne
			space_base=models[i].getMat("eigenvectors")			# eigenvectors de l'eigenspacespace de  cette personne
			space_mean=models[i].getMat("mean")				# vecteur moyen de ses images

			for j in xrange(sample_per_ID):					# Selectionne 20 images par personnes
				temp_img.append(listImagesPerson[j])					# temp_img = liste des 20 images de la personne
				temp_img[j]=temp_img[j].reshape(sz[0]*sz[1])
			print temp_img

	print "np.asarray(space_base).shape"
	print np.asarray(space_base).shape
	testing=testing.reshape(sz[0]*sz[1])					 
	space_mean=space_mean.reshape(sz[0]*sz[1])				
	
	canonic_standard_dev=functions.standard_deviation_canonic(temp_img,space_mean)  # Calcule ecart type des 7 images a leur moyenne ds 												  l'espace  canonique

	print "canonic standard deviation"
	print canonic_standard_dev # ecart type
	print "testing standard deviation"
	print functions.norm(testing-space_mean)

	proj_img = []									# projections des images en BDD dans leur eigenspace
	print "projections des images en BDD dans leur propre eigenspace"
	for i in xrange(sample_per_ID):
		proj_img.append(functions.subspace_project(space_base,space_mean, temp_img[i]))
		print proj_img[i]
	
	standard_dev=functions.standard_deviation(proj_img)				# Calcul de l'ecart type dans l'eigenspace de l'image BDD
	print "ecart type ds eigenspace "
	print standard_dev

	projection=functions.subspace_project(space_base,space_mean, testing)		# Projection image test ds eigenspace
	print "projection de l'image test"
	print projection
	print "de norme"
	print functions.norm(projection)						# calcul de sa norme

	

	reconstructed=functions.reconstruct(space_base, space_mean, projection)		# Reconstruction de l'image
	reconstructed2=functions.normalize(reconstructed,0,255,np.uint8)
	reconstructed2 = reconstructed2.reshape(sz)
	cv2.namedWindow('reconstructed', cv2.WINDOW_NORMAL)
	cv2.imshow("reconstructed", reconstructed2)
	#cv2.waitKey(0)


	# Fin partie Seuil

	# We print all the posible result
	for i in xrange(len(positiveTestResult)):
		print "Predicted label is person = %d (confidence=%.2f)" % (positiveTestResult[i][0] , positiveTestResult[i][1])

	print "best result is ( TODO)"
	#cv2.waitKey(0)


