import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]
#print images.shape
#print labels
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,random_state=1)

pca = PCA(n_components=0.8)
train_images = pca.fit_transform(train_images)
test_images = pca.transform(test_images)

forest = RandomForestClassifier(n_estimators = 500, criterion="gini", max_features=30)
forest = forest.fit(train_images,train_labels)
forest_output = forest.predict(test_images)

print(accuracy_score(test_labels, forest_output))
#0.870983099262