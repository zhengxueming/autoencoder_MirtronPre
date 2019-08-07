from model import autoencoder_model
import dataProcess
import numpy as np
from sklearn.manifold import TSNE
import os
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# process the data
FILE_PATH = "../data/miRBase_set.csv"
FILE_PATH_PUTATIVE = "../data/putative_mirtrons_set.csv"
X,y = dataProcess.generate_data(FILE_PATH,FILE_PATH_PUTATIVE)
print(X.shape)
print(y.shape)

encoder,autoencoder = autoencoder_model()
num_iteration = 1
my_optimizer = Adam(lr = 0.0003)

if os.path.exists("autoencoder_model.h5"):
        print("load the weights")
        autoencoder.load_weights("autoencoder_model.h5")

autoencoder.compile(optimizer= my_optimizer)
autoencoder.fit(X,epochs= num_iteration, batch_size=128)
autoencoder.save_weights('autoencoder_model.h5')
# store the trained encoder network
encoder.save('trained_encoder.h5')
print("trained_encoder.h5 is stored! ")


#print output of neural net
print(autoencoder.predict(X))
# same probability for the last several sequences!


# plotting
encoded_seq = encoder.predict(X)
# print(encoded_seq.shape)
# X_tsne = TSNE(n_components=3,learning_rate=100).fit_transform(encoded_seq)
X_tsne = TSNE(n_components=2,perplexity= 20,n_iter = 5000,\
             learning_rate=200).fit_transform(encoded_seq)
# print(X_train.shape)
# print(X_tsne.shape)
def get_location(x,y):
    """ find the location of x in y
    
    # args:
        x: np.array
        y: np.array,one more dimension than x
    # returns:
        a list of location where the elements in y is exactly the x
    
    """
    location_list = []
    m_len = len(y)
    for i in range(m_len):
        if x.tolist() == y.tolist()[i]:
            location_list.append(i)
    return location_list
            
mirtrons_loacation = get_location(np.array([1,0]),y)
print("mirtrons_num:{}".format(len(mirtrons_loacation)))

positive_X_tsne = X_tsne[mirtrons_loacation]     # Mirtrons
negative_X_tsne = X_tsne[list(set(i for i in range(len(X_tsne)))- set(mirtrons_loacation))]   # canonical microRN

plt.scatter(positive_X_tsne[:,0], positive_X_tsne[:, 1],s=15,label='Mirtrons',marker = "o")
plt.scatter(negative_X_tsne[:, 0], negative_X_tsne[:, 1],s=15,label= 'Canonical',marker = "x")
plt.legend(loc = 'upper right')
plt.show()
