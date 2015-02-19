#Evaluates the model, given training set, known test set & unknown test set
#Saves data into a file

import classes
import numpy as np
import cv2
import functions
import models

normalize = True
    
resultsglobal = []
resultsperperson = []
train = functions.read_images('data/trainingNew/',(256,256))
test1 = functions.read_images('data/testingNew/known_persons/',(256,256))
test2 = functions.read_images('data/testingNew/unknown_persons/',(256,256))

if normalize :
	for i in range(len(train[0])):
		cv2.equalizeHist(train[0][i], train[0][i])
	for i in range(len(test1[0])):
		cv2.equalizeHist(test1[0][i], test1[0][i])
	for i in range(len(test2[0])):
		cv2.equalizeHist(test2[0][i], test2[0][i])
	print "Finished Normalizing images"

for eigen in range(3,10) :
	for threshold in range(0,20) :
		#modelglobal=models.ModelPCAGlobal(eigen,threshold*30,None, (256,256), train )
		modelperperson=models.ModelPerPerson(eigen,threshold*300,None, (256,256), train )
		#resultsglobal.append(modelglobal.evaluate(None,None,test1, test2))
		resultsperperson.append(modelperperson.evaluate(None,None,test1, test2))
		print threshold
		print eigen
		print '\n'
#np.savetxt('global_eigen_3-10_threshold_0-6000',np.asarray(resultsglobal))
np.savetxt('new_base_normalized_per-person_eigen_0-10_threshold_0-6000',np.asarray(resultsperperson))





