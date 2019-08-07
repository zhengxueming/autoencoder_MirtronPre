#! /usr/bin/env python3

"""Predict mirtron with probability using learned latent variables. "yes" means "is a    mirtron"(probability >=0.8) ."no" means "is not a mirtron"(probability <=0.2). "not sure" (0.2 < probability < 0.8)
"""

from sklearn import svm
import sys
sys.path.append("./autoencoder")
import dataProcess
import numpy as np
from keras.models import load_model


x_cast = {"A":[1,0,0,0],"U":[0,1,0,0],\
              "T":[0,1,0,0],"G":[0,0,1,0],\
              "C":[0,0,0,1],"N":[0,0,0,0]}
y_cast = {"TRUE": 1,"FALSE":-1} #TRUE:Mirtrons  FALSE:canonical microRN


def usage():
    print("USAGE: python isMirtron.py -s pre-miRNA sequnece\n")
    print("Example: python isMirtron.py -s CTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG")


# padding and trim the sequence to the length of SEQUENCE_LENGTH
SEQUENCE_LENGTH = 164
def seq_process(seq):
    # remove all the spaces
    seq = seq.replace(' ', '')
    # remove \n
    seq = seq.replace("\n", "")
    for base in seq:
        if base not in ("A","U","G","C","T","a","u","g","c","t"):
            print ("Input sequence is wrong: incorrect base!\n")
            usage()
            exit(1)
    seq = seq.upper() 
    m_len = len(seq)
    if m_len < SEQUENCE_LENGTH:
        seq += "N" *(SEQUENCE_LENGTH - m_len)
    elif m_len >= SEQUENCE_LENGTH:
        seq = seq[:SEQUENCE_LENGTH]

    return seq


# sequence vectorization
def seq_vectorize(seq):
    vectorized_seq = []
    for char in processed_seq:
        vectorized_seq.append(x_cast[char])
    vectorized_seq = np.array(vectorized_seq)
    vectorized_seq = vectorized_seq.reshape([1,SEQUENCE_LENGTH,4]) 
    return vectorized_seq


# get the parameters 
def get_input_seq():
    import getopt
    input_seq = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hs:",["help","sequence="])
    except getopt.GetoptError:
        print ("Wrong usage!\n")
        usage()
        sys.exit(1)
    if len(opts) < 1:
        usage()
        sys.exit(1)
    # parse the options
    for op, value in opts:
        if op in ("-s","--sequence"):
            input_seq = value
        elif op in ("-h","--help"):
            usage()
            sys.exit()
    return input_seq


# handle the input and get the input sequence 
input_seq = get_input_seq()
# check the input sequence
processed_seq = seq_process(input_seq)
# vectorized the input sequence with one-hot encoding
vectorized_seq = seq_vectorize(processed_seq)

FILE_PATH = "./data/miRBase_set.csv" 
FILE_PATH_PUTATIVE = "./data/putative_mirtrons_set.csv"

X,y = dataProcess.generate_data(FILE_PATH,FILE_PATH_PUTATIVE)

# convert the value of y for svm model
y = y.tolist()
for i in range(len(y)):
    if y[i] ==[1,0]:
        y[i] = 1
    if y[i] ==[0,1]:
        y[i] = -1
y = np.array(y)
#print(X.shape)
#print(y.shape)
#print("===============")


# load the trained deep autoencoder model
encoder_model = load_model('./autoencoder/trained_encoder.h5')
# encoding the X in the dataset
X_encoded = encoder_model.predict(X)
# encoding the vectorized input sequence 
vectorized_seq_encoded = encoder_model.predict(vectorized_seq)

# establish the SVC model
svm_model = svm.SVC(kernel='rbf', C=5,gamma = 0.0001,probability=True)
# train the model
svm_model.fit(X_encoded, y)

# prediction results (probability)
result_probability = svm_model.predict_proba(vectorized_seq_encoded)
# print(result_probability)
result_probability = result_probability[0][1]    # probability of mirtron
# print the prediction result
up_threhold = 0.8
down_threhold = 0.5
print("==================================================================")
print("Prediction result:")
if result_probability >= up_threhold:
    print("Yes,it is a mirtron.")
    print("probability:{}".format(result_probability))
elif result_probability <= down_threhold:
    print("No,it is not a mirtron.")
else:
    print("Not sure.")
print("\n")




