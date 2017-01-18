import sys
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

if len(sys.argv) > 1:
    filename = open(sys.argv[1])
else:
    filename = 'wrong.data'

def apply_pca(train_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	return train_images

print("Reading Data from..",filename)
testing_data = pd.read_csv(filename, delim_whitespace=True, header=None)
testing_images = testing_data.iloc[:,2:]
testing_labels = testing_data.iloc[:,0]
print("Done..!!")

print("Applying Dimension Reduction..!!")
testing_images = apply_pca(testing_images, 37)

print("Loading Classifier..!!")
clf = joblib.load('abhishek.pkl')
print("Loading Successfull..!!")

print("Predicting Values..!!")
predicted_labels = clf.predict(testing_images)

print("Prediction Complete...Writing into a file..!!")
np.savetxt('predicted_values.txt', np.c_[predicted_labels], delimiter=' ', header = 'class')
print("Writing data complete..!!")