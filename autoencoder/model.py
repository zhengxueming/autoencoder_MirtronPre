""" construct the autoencoder model
"""

import keras
from keras import regularizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Activation,Dropout,Flatten,Input,Reshape
from keras.layers import Conv1D,MaxPooling1D,UpSampling1D
from keras import backend as K
from keras.losses import binary_crossentropy


def autoencoder_model():
   """ design the autoencoder structure.
   """
   # the dimension of latent variable
   encoding_dim = 10 
   # input placeholder 
   input_data = Input(shape=(164,4))

   # add noise

   # coding layers 
   encoded = Conv1D(32,2, activation='relu',padding = 'same')(input_data)
   encoded = MaxPooling1D(pool_size = 2)(encoded)
   encoded = Conv1D(64,3, activation='relu',padding = 'same')(encoded)
   encoded = MaxPooling1D(pool_size = 2)(encoded)
   encoded = Conv1D(128,4, activation='relu',padding = 'same')(encoded)

   encoded = Flatten()(encoded)
   # encoder = Model(inputs=input_data, outputs=encoded) 
   # print(encoder.summary())

   encoded = Dense(128, activation='relu',\
                    kernel_regularizer=regularizers.l2(0.01))(encoded)
   encoder_output = Dense(encoding_dim, activation='relu')(encoded) 

   # encoder = Model(inputs=input_data, outputs=encoder_output) 
   # print(encoder.summary())

   # decoding layers 
   decoded = Dense(128, activation='relu')(encoder_output)
   decoded = Dense(5248, activation='relu',\
                   kernel_regularizer=regularizers.l2(0.01))(decoded)
   decoded = Reshape((-1,128))(decoded)

   decoded = Conv1D(64,3,activation='relu',padding = 'same')(decoded)
   decoded = UpSampling1D(size = 2)(decoded)

   decoded = Conv1D(32,2,activation='relu',padding = 'same')(decoded)
   decoded = UpSampling1D(size = 2)(decoded)
   decoded_output = Conv1D(4,2,activation='softmax',padding = 'same')(decoded)

   # encoder model 
   encoder = Model(inputs=input_data, outputs=encoder_output)
   print(encoder.summary())

   # autoencoder model
   autoencoder = Model(inputs=input_data, outputs=decoded_output) 
   print(autoencoder.summary())
   autoencoder_loss = binary_crossentropy\
                       (K.flatten(input_data),K.flatten(decoded_output)) 
   autoencoder.add_loss(autoencoder_loss)   # add the loss to the model

   return encoder,autoencoder


if __name__ == "__main__":
   encoder,autoencoder = autoencoder_model()
   print("model constructed!")


