#Functions
import os
import sys
import cv2
import numpy as np
import math
import PyQt4 as qt
from PyQt4 import QtGui


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
	    (head, folder_name) = os.path.split(subject_path)
	    parts = folder_name.split("_")
	    label = int(parts[len(parts)-1])
		
            for filename in os.listdir(subject_path):
                if filename != ".DS_Store" and filename != "._.DS_Store":
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                        # resize to given size (if given)
                        if (sz is not None):
                            im = cv2.resize(im, sz)
                        else:
                            im = cv2.resize(im, (256,256))
                        #im = cv2.resize(im, (76,98))
                        #cv2.resize(im, im, (64, 64), 0, 0, cv2.INTER_CUBIC);
                        if (im is not None):
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(label)
                    except IOError, (errno, strerror):
                        print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
    y=np.asarray(y, dtype=np.int32)	
    return [X,y]

# Return a table containing one model per person 
def divideModelsByPerson(X,y):
    models = []
    [Xtmp,ytmp] = [],[]
    unique_labels=np.unique(y)
 
    for j in unique_labels:
        for i in xrange(len(X)):
            if y[i] == j :
                Xtmp.append(X[i])
                ytmp.append(y[i])
        models.append([Xtmp,ytmp])
        [Xtmp,ytmp] = [],[]
    return np.asarray(models)


def subspace_project(eigenvectors_column, mean, source):
    delta_from_mean = (source-mean)
    delta_from_mean = delta_from_mean.flatten()
    result = np.dot(delta_from_mean,eigenvectors_column)
    return result

def reconstruct(eigenVectors_column, mean, coordinates):
    i=0
    reconstructed=np.zeros_like(mean)
    while i < len(coordinates):
        a = coordinates[i] * eigenVectors_column[:,i]
        reconstructed  = reconstructed + a
        i = i + 1
    reconstructed = reconstructed + mean
    return reconstructed

#Tristan


def standard_deviation(sample_vectors):
	variance=0
	card=len(sample_vectors)
	norms = []
	for i in xrange(card):
		norms.append(norm(sample_vectors[i]))
		variance=variance+norms[i]**2
		
	variance=variance/card
	result=math.sqrt(variance)
	
	return result

def standard_deviation_canonic(sample_vectors, mean_vector):
	variance=0
	card=len(sample_vectors)
	for i in xrange(card):
		currentNorm = norm(sample_vectors[i]-mean_vector) # le soucis vient peut etre d'ici
		variance=variance+(currentNorm**2)
		
	variance=variance/card
	result=math.sqrt(variance) # result = ecart type 

	return result
	
def norm(vector):
	result=0
	for i in xrange(len(vector)):
		result=result+vector[i]*vector[i]
	result=math.sqrt(result)
	return result
	

def error_matrix(base_vector, mean_vector):
	result=0
	return result

def getNumberOfPersonInFolder(pathToFolder):
    i = 0
    j = 0
    for dirname, dirnames, filenames in os.walk(pathToFolder):
        i = i+1
        for subdirname in dirnames:
            j=j+1
    return j

def project(vec1, vec2):
	sum=0
	if len(vec1)!=len(vec2):
		print "function project incompatible vec dimensions"
	else:
		for i in xrange(len(vec2)):
			sum=sum+vec1[i]*vec2[i]
	return sum

def retreiveAnImageFromFolder(pathToData, labelUser):
    for dirname, dirnames, filenames in os.walk(pathToData):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            (head, folder_name) = os.path.split(subject_path)
            parts = folder_name.split("_")
            label = int(parts[len(parts)-1])
            if (label == labelUser ):  
                #return parts[0]
                for filename in os.listdir(subject_path):
                                if filename != ".DS_Store":
                                    try:
                                        im = QtGui.QPixmap(os.path.join(subject_path, filename))
                                        return im

                                    except IOError, (errno, strerror):
                                        print "I/O error({0}): {1}".format(errno, strerror)
                                    except:
                                        print "Unexpected error:", sys.exc_info()[0]
                                        raise
                                    break
    print os.getcwd()+"data/not_found.png"
    im = QtGui.QPixmap(os.getcwd()+"/data/not_found.jpg")
    return im


# Faire attention a la correspondance des models et modelsData
def probaCorrectRecognition(person_ID, modelsData, models, testingVec, sz):
    

    #parameters 
    nbrOfEigenvectorToRemove=15	# Attention a ne pas fixer une valeur superieur au nombre de photos par visage 	(environ egale a 10)
    eigenSpaceDimension = 0		# Valeur attribuee plus tard (ligne X)
    sample_per_ID=len(listImagesPerson)

    temp_img = []
    for i in xrange(len(modelsData)):
        label=modelsData[i][1][0]   						# label des personnes de la base
        if label==person_ID: 							# S'il s'agit de la personne reconnue
            listImagesPerson=modelsData[i][0]				# liste d'image pour une meme personne
            space_base=models[i].getMat("eigenvectors")			# eigenvectors de l'eigenspacespace de  cette personne
            space_mean=models[i].getMat("mean")				# vecteur moyen de ses images
            eigenvalues=models[i].getMat("eigenvalues")
            eigenSpaceDimension = len(modelsData[i][1])

            for j in xrange(len(listImagesPerson)):					# Selectionne 20 images par personnes
                temp_img.append(listImagesPerson[j])			# temp_img = liste des 20 images de la personne
                temp_img[j]=temp_img[j].reshape(sz[0]*sz[1])		# print temp_img


    testingVec=testingVec.reshape(sz[0]*sz[1])					 
    space_mean=space_mean.reshape(sz[0]*sz[1])

    canonic_standard_dev=standard_deviation_canonic(temp_img,space_mean)
    ratio=norm(testingVec-space_mean)/canonic_standard_dev

    return ratio


def probaCorrectRecognition_Projected(person_ID, modelsData, models, testingVec,sz,nbrOfEigenvectorToRemove):
    #parameters 
     # Valeur attribuee plus tard (ligne X)
    sample_per_ID=25

    temp_img = []
    for i in xrange(len(modelsData)):
        label=modelsData[i][1][0]                           # label des personnes de la base
        if label==person_ID:                            # S'il s'agit de la personne reconnue
            listImagesPerson=modelsData[i][0]               # liste d'image pour une meme personne
            space_base=models[i].getMat("eigenvectors")         # eigenvectors de l'eigenspacespace de  cette personne
            space_mean=models[i].getMat("mean")             # vecteur moyen de ses images
            eigenvalues=models[i].getMat("eigenvalues")
            eigenSpaceDimension = len(modelsData[i][1])

            for j in xrange(len(listImagesPerson)):                 # Selectionne 20 images par personnes
                temp_img.append(listImagesPerson[j])            # temp_img = liste des 20 images de la personne
                temp_img[j]=temp_img[j].reshape(sz[0]*sz[1])        # print temp_img
    space_mean=space_mean.reshape(sz[0]*sz[1])
    testingVec=testingVec.reshape(sz[0]*sz[1])
    optimal_subspace_base = []
    print "space-base shape"
    print np.asarray(space_base).shape
    print "Index checking"
    if (eigenSpaceDimension==sample_per_ID):
        dim=eigenSpaceDimension-nbrOfEigenvectorToRemove
        for i in xrange(dim): 
            optimal_subspace_base.append(space_base[:,i+nbrOfEigenvectorToRemove]) 
    else:
        dim=eigenSpaceDimension
        for i in xrange(dim): 
            optimal_subspace_base.append(space_base[:,i]) 

    optimal_subspace_base=zip(*np.asarray(optimal_subspace_base))
    proj_img_optimal = []                   # Projection du nuage de point de la classe ds son espace optimise

    for i in xrange(len(listImagesPerson)):
        proj_img_optimal.append(subspace_project(optimal_subspace_base,space_mean, temp_img[i]))

    projection_optimal=subspace_project(optimal_subspace_base,space_mean, testingVec)    # Projection image test ds eigenspace
    standard_dev_optimal=standard_deviation(proj_img_optimal)     # Calcul de l'ecart type dans l'eigenspace de l'image BDD
    ratio=norm(projection_optimal)/standard_dev_optimal
    return ratio
