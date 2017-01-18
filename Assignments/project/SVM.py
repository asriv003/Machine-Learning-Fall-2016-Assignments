import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,GridSearchCV,StratifiedShuffleSplit
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np

labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]

#print images.shape
#print labels

def plot_learning_curve(estimator):
	plt.figure()
	plt.title("Learning SVM Curves")
	plt.ylim(0.7,1.01)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, images, labels)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
	plt.legend(loc="best")
	plt.show()

def grid_search_cv():
	pca = PCA(n_components=0.7)
	temp_images = pca.fit_transform(images)
	C_range = np.linspace(1, 100,10)
	gamma_range = [0.0001,0.001,0.01,0.1,1]
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
	grid.fit(temp_images, labels)
	return grid.best_params_


def apply_pca(train_images,test_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	test_images = pca.transform(test_images)
	print train_images.shape
	print test_images.shape
	return train_images,test_images

print np.logspace(-3, 2, 10)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7,random_state=1)
C_grid=10
gamma_grid='auto'
print gamma_grid
#best_param = grid_search_cv()
#C_grid = best_param.get('C')
#gamma_grid = best_param.get('gamma')
clf = svm.SVC(kernel='rbf',C=C_grid,gamma=gamma_grid)
#plot_learning_curve(clf)
train_images,test_images = apply_pca(train_images,test_images,0.7)
clf.fit(train_images, train_labels)
cvs = cross_val_score(clf, train_images, train_labels,scoring='precision_macro')
print np.mean(cvs)
svm_output = clf.predict(test_images)
print clf.score(test_images,test_labels)
#print confusion_matrix(test_labels, svm_output)