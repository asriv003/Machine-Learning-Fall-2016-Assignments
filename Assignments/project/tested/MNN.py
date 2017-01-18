import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,0]
#print images.shape
#print labels
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7,random_state=1)
#i=1
#img = train_images.iloc[i].as_matrix()
# img = img.reshape(16,8)
# plt.imshow(img,cmap='gray')
# plt.title(train_labels.iloc[i,0])
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, activation="tanh")
clf.fit(train_images, train_labels)
print clf.score(test_images,test_labels)
#0.835991430612
