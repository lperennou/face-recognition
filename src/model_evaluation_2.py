#Prints visual representation of the data saved in model_evaluation.py
#prints eigenvectors in X, threshold in Y and one of the following in Z : false negatives, false positives, missclassifications, total error

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt('new_base_normalized_per-person_eigen_0-10_threshold_0-6000')
data_final = []
for eigen in range(3,10) :
	for threshold in range(0,20) :
		data_final.append([eigen, threshold * 300])
data_final = np.asarray(data_final)
		

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(data_final[:,0],data_final[:,1], data[:,0])
plt.xlabel('Dimension')
plt.ylabel('Threshold')
plt.title('False Positive')

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(data_final[:,0],data_final[:,1], data[:,1])
plt.xlabel('Dimension')
plt.ylabel('Threshold')
plt.title('False Negative')

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(data_final[:,0],data_final[:,1], data[:,2])
plt.xlabel('Dimension')
plt.ylabel('Threshold')
plt.title('Missclassifications')

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(data_final[:,0],data_final[:,1], data[:,3])
plt.xlabel('Dimension')
plt.ylabel('Threshold')
plt.title('Total Error')

plt.show()

