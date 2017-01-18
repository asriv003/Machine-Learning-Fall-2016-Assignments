import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]
#print images.shape
#print labels
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,random_state=1)

forest = AdaBoostClassifier(n_estimators = 500)
forest = forest.fit(train_images,train_labels)
forest_output = forest.predict(test_images)
#print cross_val_score(forest, train_images, train_labels)
print(1 - accuracy_score(test_labels, forest_output))
#0.790983099262