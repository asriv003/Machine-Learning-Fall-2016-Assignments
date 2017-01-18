import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# The competition datafiles are in the directory ../input
# Read competition data files:
labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]
#print images.shape
#print labels
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,random_state=1)

pca = PCA(n_components=0.8)
train_images = pca.fit_transform(train_images)
test_images = pca.transform(test_images)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_images, train_labels)

knn_output = neigh.predict(test_images)
print(accuracy_score(test_labels, knn_output))
#pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)