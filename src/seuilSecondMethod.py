#!/usr/bin/python
import numpy as np
import cv2
import cv
import uuid
import functions
import os
import sys

sz=(64,64)
numberOfEigenfaces = 50
topEigenValues= 10
nbrOfEigenvectorToRemove=15				# Attention a ne pas fixer une valeur superieur au nombre de photos par visage (environ egale a 10)
eigenSpaceDimension = 0					# Valeur attribuee ligne 142
threshold = 5000

sample_per_ID=25
person_ID= 11

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

	temp_img = []

	for i in xrange(len(modelsData)):
		label=modelsData[i][1][0]   							# label des personnes de la base
		if label==person_ID: 								# S'il s'agit de la personne reconnue
			listImagesPerson=modelsData[i][0]					# liste d'image pour une meme personne
			space_base=models[i].getMat("eigenvectors")				# eigenvectors de l'eigenspacespace de  cette personne
			space_mean=models[i].getMat("mean")					# vecteur moyen de ses images
			eigenvalues=models[i].getMat("eigenvalues")
			

			for j in xrange(sample_per_ID):						# Selectionne 20 images par personnes
				temp_img.append(listImagesPerson[j])				# temp_img = liste des 20 images de la personne
				temp_img[j]=temp_img[j].reshape(sz[0]*sz[1])			#print temp_img


	# Calcul du rapport R = (ecart type / norme vecteur error de test image ) , dans l'espace canonique
	testing=testing.reshape(sz[0]*sz[1])					 
	space_mean=space_mean.reshape(sz[0]*sz[1])	
	print " --------  canonic space   ---------- "			
	canonic_standard_dev=functions.standard_deviation_canonic(temp_img,space_mean)  # Calcule ecart type des 7 images a leur moyenne ds l'espace  canonique
	print "canonic standard deviation"
	print canonic_standard_dev # ecart type
	print "testing standard deviation"
	print functions.norm(testing-space_mean)
	print "ratio"
	print functions.norm(testing-space_mean)/canonic_standard_dev

	print " --------  canonic space second computation  ---------- "
	tryresult=functions.probaCorrectRecognition(person_ID, modelsData, models, testing, sz)
	print "Ratio"
	print tryresult
	# Calcul du rapport dans l'eigenspace (maximise variance=pire ratio)


	unoptimal_subspace_base = []
	print " --------  Unoptimised space   ---------- "	
	print "unoptimal space base size"
	print topEigenValues
	
	for i in xrange(topEigenValues):
		unoptimal_subspace_base.append(space_base[:,i]) 
	unoptimal_subspace_base=zip(*np.asarray(unoptimal_subspace_base))
	proj_img_unoptimal = []					# Projection du nuage de point de la classe ds son espace non optimise
	

	for i in xrange(sample_per_ID):
		proj_img_unoptimal.append(functions.subspace_project(unoptimal_subspace_base,space_mean, temp_img[i]))

	projection_unoptimal=functions.subspace_project(unoptimal_subspace_base,space_mean, testing)	# Projection image test ds eigenspace
	standard_dev_unoptimal=functions.standard_deviation(proj_img_unoptimal)				# Calcul de l'ecart type dans l'eigenspace de l'image BDD

	print "standard deviation"
	print standard_dev_unoptimal # ecart type
	print "testing standard deviation"
	print functions.norm(projection_unoptimal)
	print "ratio"
	print functions.norm(projection_unoptimal)/standard_dev_unoptimal



	# Calcul du rapport dans l'espace optimise (minimise variance=meilleur ratio)	
 	# We remove the main eigensvectors to keep the high variance more
 	eigenSpaceDimension=len(modelsData[person_ID][0])
	eigenSpaceDimension=25
 	print "eigenSpaceDimension is "
 	print eigenSpaceDimension
	optimal_subspace_base = []
	
	for i in xrange(eigenSpaceDimension-nbrOfEigenvectorToRemove):
		optimal_subspace_base.append(space_base[:,i+nbrOfEigenvectorToRemove]) 
	optimal_subspace_base=zip(*np.asarray(optimal_subspace_base))
	proj_img_optimal = []					# Projection du nuage de point de la classe ds son espace optimise

	for i in xrange(sample_per_ID):
		proj_img_optimal.append(functions.subspace_project(optimal_subspace_base,space_mean, temp_img[i]))

	projection_optimal=functions.subspace_project(optimal_subspace_base,space_mean, testing)	# Projection image test ds eigenspace
	standard_dev_optimal=functions.standard_deviation(proj_img_optimal)		# Calcul de l'ecart type dans l'eigenspace de l'image BDD


	print " --------  Optimised space   ---------- "
	print "Optimal space base size"	
	print eigenSpaceDimension-nbrOfEigenvectorToRemove
	print "standard deviation"
	print standard_dev_optimal # ecart type
	print "testing standard deviation"
	print functions.norm(projection_optimal)
	print "ratio"
	print functions.norm(projection_optimal)/standard_dev_optimal
	
#	print " --------  checking bases   ---------- "
#	print " space_base "
#	print np.asarray(space_base)
#	print " unoptimal_subspace_base "
#	print  np.asarray(unoptimal_subspace_base) 
#	print " optimal_subspace_base "
#	print  np.asarray(optimal_subspace_base)


	# Fin partie Seuil

	# We print all the posible result
	for i in xrange(len(positiveTestResult)):
		print "Predicted label is person = %d (confidence=%.2f)" % (positiveTestResult[i][0] , positiveTestResult[i][1])

	print "best result is ( TODO)"
	#cv2.waitKey(0)


