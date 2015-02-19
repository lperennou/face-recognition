#!/usr/bin/python
import numpy as np
import cv2
import cv
import uuid
import functions
import os
import sys

sz=(100,100)
numberOfEigenfaces = 5
threshold = 3000

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: recognition.py </path/to/training/images> </path/to/testing/image.jpg>"
		sys.exit()
	# Read the images that has been extract 
	[X,y] = functions.read_images(sys.argv[1],sz)
	WindowsName="img"	
	cv2.namedWindow(WindowsName, cv2.WINDOW_NORMAL)	
	print "Nbr of image "
	print len(X)
	print  "Nbr of person "
	print len(y)
	# Convert labels to 32bit integers. This is a workaround for 64bit machines
	y = np.asarray(y, dtype=np.int32)
	out_dir = "tmp/pca"
	# Create the Eigenfaces model. We are going to use the default
	# parameters for this simple example, please read the documentation
	# for thresholding:
	model = cv2.createEigenFaceRecognizer(numberOfEigenfaces, threshold)
	# Read
	# Learn the model. Remember our function returns Python lists,
	# so we use np.asarray to turn them into NumPy lists to make
	# the OpenCV wrapper happy:
	model.train(np.asarray(X), np.asarray(y))
	# We now get a prediction from the model! In reality you
	# should always use unseen images for testing your model.
	# But so many people were confused, when I sliced an image
	# off in the C++ version, so I am just using an image we
	# have trained with.
	#
	# model.predict is going to return the predicted label and
	# the associated confidence:
	# Here we choose our test image 
	testing = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
	#test = []
	if (sz is not None ):
		testing = cv2.resize(testing, sz)
	#else:
		#testing = cv2.resize(testing, (256,256))
	cv2.imshow("img",testing)

		
	[p_label, p_confidence] = model.predict(testing)
	# Print it:
	print "Predicted label is person = %d (confidence=%.2f)" % (p_label , p_confidence)
	# Cool! Finally we'll plot the Eigenfaces, because that's
	# what most people read in the papers are keen to see.
	#
	# Just like in C++ you have access to all model internal
	# data, because the cv::FaceRecognizer is a cv::Algorithm.
	#
	# You can see the available parameters with getParams():
	print model.getParams()
	# Now let's get some data:
	mean = model.getMat("mean")
	mean=mean.reshape(sz[0]*sz[1])
	eigenVectors = model.getMat("eigenvectors")
	testing=testing.reshape(sz[0]*sz[1])

	projection=functions.subspace_project(eigenVectors,mean, testing)
	print testing.shape
	print mean.shape
	print eigenVectors.shape

	reconstructed=functions.reconstruct(eigenVectors, mean, projection)
	reconstructed2=functions.normalize(reconstructed,0,255,np.uint8)
	reconstructed2 = reconstructed2.reshape(sz)
	
	cv2.namedWindow('reconstructed', cv2.WINDOW_NORMAL)
	cv2.imshow("reconstructed", reconstructed2)
	cv2.waitKey(0)



	eigenvectors = model.getMat("eigenvectors")
	# We'll save the mean, by first normalizing it:
	mean_norm = cv2.normalize(mean, 0, 255, dtype=np.uint8)
	mean_resized = mean_norm.reshape(X[0].shape)
	if out_dir is None:
	    cv2.imshow("mean", mean_resized)
	else:
	    cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
	# Turn the first (at most) 16 eigenvectors into grayscale
	# images. You could also use cv::cv2.normalize here, but sticking
	# to NumPy is much easier for now.
	# Note: eigenvectors are stored by column:
	for i in xrange(min(len(X), 16)):
	    eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
	    eigenvector_i_norm = cv2.normalize(eigenvector_i, 0, 255, dtype=np.uint8)
	    # Show or save the images:
	    if out_dir is None:
	        cv2.imshow("%s/eigenface_%d" % (out_dir,i), eigenvector_i_norm)
	    else:
	        cv2.imwrite("%s/eigenface_%d.png" % (out_dir,i), eigenvector_i_norm)
	# Show the images:
	if out_dir is None:
	    cv2.waitKey(0)


