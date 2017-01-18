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
    filename = 'handwriting.data'

print("Reading the data..from ",filename)

labeled_images = pd.read_csv(filename, delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]

def apply_pca(train_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	return train_images

print("Declaring Classifier..!!")
clf = svm.SVC(kernel='rbf',C=10,gamma=0.1)
print("Applying Dimension Reduction..!!")
images = apply_pca(images,0.7)
print("Training data..!!!")
clf.fit(images, labels)
print("Saving Classifier..!!")
joblib.dump(clf, 'abhishek.pkl')