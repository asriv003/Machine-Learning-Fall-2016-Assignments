import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

filename1 = 'predicted_values.txt'

if len(sys.argv) > 1:
    filename2 = open(sys.argv[1])
else:
    filename2 = 'wrong.data'

print("Reading Data..!!")
predicted_data = pd.read_csv(filename1, delim_whitespace=True)
predicted_Y = predicted_data.iloc[:,0]

#print predicted_Y.shape

true_data = pd.read_csv(filename2, delim_whitespace=True, header=None)
true_Y = true_data.iloc[:,0]
#print true_Y.shape

print("Accuracy: ", accuracy_score(true_Y, predicted_Y))