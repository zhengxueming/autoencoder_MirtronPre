# -*- coding: utf-8 -*-
"""data process module. 
"""

import numpy as np
import pandas as pd
import csv
import random

FILE_PATH = "../data/miRBase_set.csv" 
FILE_PATH_PUTATIVE = "../data/putative_mirtrons_set.csv"


# read csv file and generate the train and test data set
def generate_data(file1_path,file2_path):
    csv_reader=csv.reader(open(file1_path, encoding='utf-8'))
    csv_reader_putative=csv.reader(open(file2_path, encoding='utf-8'))

    all_data_set = []
    tansformed_all_data_set = []
    X = []
    y = []
    
    # read the data into a list(name,sequence,class)
    for row in csv_reader:
        all_data_set.append([row[0],row[1],row[2]])
    for row in csv_reader_putative:
        all_data_set.append([row[0],row[1],row[2]])    
    # shuffle the data set randomly    
    random.seed(2)
    random.shuffle(all_data_set)

    # get the maxmium length of the seqence
    max_seq_len = 0
    for item in all_data_set:
        if len(item[1])>max_seq_len:
            max_seq_len = len(item[1])
    # print(max_seq_len)
    
    # padding to max_seq_len
    for item in all_data_set:
        item[1] += "N" *(max_seq_len-len(item[1]))

    x_cast = {"A":[1,0,0,0],"U":[0,1,0,0],\
              "T":[0,1,0,0],"G":[0,0,1,0],\
              "C":[0,0,0,1],"N":[0,0,0,0]}
    y_cast = {"TRUE": [1,0],"FALSE":[0,1]} #TRUE:Mirtrons  FALSE:canonical microRN
    
    for item in all_data_set:
        data = []
        for char in item[1]:
            data.append(x_cast[char])
        tansformed_all_data_set.append([item[0],data,y_cast[item[2]]])
        X.append(data)
        y.append(y_cast[item[2]])
    #dataframe = pd.DataFrame(tansformed_all_data_set,\
    #                            columns = ['name','seq_vec','class'])
    return np.array(X),np.array(y)

if __name__ == "__main__":
   X,y = generate_data(FILE_PATH,FILE_PATH_PUTATIVE)
   # print(X)
   print(X.shape)
   # print(y)
   print(y.shape)
   print("data are ready")
