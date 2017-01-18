import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]
#print images.shape
#print labels
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,random_state=0)

clf = MLPClassifier(solver='lbfgs', random_state=1)
clf.fit(train_images, train_labels)
neural_output = clf.predict(test_images)
print(accuracy_score(test_labels, neural_output))
#0.858486074744