"""" train the svm classifier with encoded latent variables.
     evaluate the performance on the test dataset
"""

import sys
sys.path.append("./autoencoder")
import dataProcess
import numpy as np

#from keras.models import Sequential,Model,load_model
from keras.models import load_model
from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
import math

FILE_PATH = "./data/miRBase_set.csv" 
FILE_PATH_PUTATIVE = "./data/putative_mirtrons_set.csv"

X,y = dataProcess.generate_data(FILE_PATH,FILE_PATH_PUTATIVE)
print(X.shape)
print(y.shape)
print("===============")

# svm_model = svm.LinearSVC()
# svm_model = svm.SVC(kernel='rbf', C=1, gamma=1)
encoder_model = load_model('./autoencoder/trained_encoder.h5')
X_encoded = encoder_model.predict(X)

y = y.tolist()

for i in range(len(y)):
    if y[i] ==[1,0]:
        y[i] = 1
    if y[i] ==[0,1]:
        y[i] = -1
y = np.array(y)
print(y.shape)


from sklearn.model_selection import GridSearchCV
parameters = {
 'C': range(1,10),
 'gamma': [0.000001,0.00001,0.00005,0.0001, 0.001, 0.1, 1, 10],
 'kernel': ['rbf']
}

svc = svm.SVC()
svm_model = GridSearchCV(svc, parameters, cv=10, n_jobs=2)


    

