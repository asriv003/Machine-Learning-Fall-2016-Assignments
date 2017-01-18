import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout

from sklearn.model_selection import train_test_split


#load data
labeled_images = pd.read_csv('handwriting.data', delim_whitespace=True, header=None)

#fix random seed for reproductibility
seed = 7
np.random.seed(seed)

#extract images pixels
images = labeled_images.iloc[0:1000,2:]
num_pixels = images.shape[1]
print num_pixels
print images.shape
#extract alphabets labels
labels = labeled_images.iloc[0:1000,0]
labels = np_utils.to_categorical(labels)
num_classes = labels.shape[1]
print num_classes
#model

def mlp_model():
	#create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	#compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,random_state=0)

#build the model
model = mlp_model()

#fit the model
#model.fit(train_images, train_labels, nb_epoch=10, batch_size=200,verbose=2)
model.train_on_batch(train_images, train_labels)
#evaluate
#score = model.evaluate(test_images, test_labels)
# use the NN model to classify test data
pred = model.predict(test_images)
pred = pred.argmax(1) # transform the binary matrix to an array of 0 to 9 labels
print pred
