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


# calculate TP,TN, FP and FN
def predict_comparision(y_test_hat,y_test):
    #tp,tn,fp,fn = 0.00001,0.00001,0.000001,0.000001
    tp,tn,fp,fn = 0,0,0,0
    m = len(y_test)
    for i in range(m):
        if y_test_hat[i] == 1:
            if y_test[i]== 1:
                tp +=1    
            else:
                fp +=1
        else:
            if y_test[i]== -1:
                tn +=1
            else:
                fn += 1 
    return tp,tn,fp,fn

# calculate SENS, SPEC, ACC and Matthews Correlation Coefficient (MCC)
def calculate_performance(tp,tn,fp,fn):
    # calculate sens, spec, f1,mcc and acc based on tp,tn,fp,fn   
    try:
        sensitivity = float(tp)/(float(tp) + float(fn))
        specifity = float(tn)/(float(tn) + float(fp))
        precision = float(tp)/(float(tp) + float(fp))
        accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + \
                                            float(fn) + float(tn))
        f1_score = (2 * (precision * sensitivity)) / (precision + sensitivity)
        mcc = ((float(tp) * float(tn)) - (float(fp) * float(fn))) /\
                math.sqrt((float(tp) + float(fp)) * (float(tp) + float(fn))*\
                (float(tn) + float(fp)) * (float(tn) + float(fn)))
    except ZeroDivisionError as err:
        print("Exception:",err)
        exit(1)
    print("Sensitivity/recall on the test data is :{}".format(sensitivity)) 
    print("specifity on the test data is :{}".format(specifity)) 
    print("precision on the test data is :{}".format(precision))
    print("accuracy on the test data is :{}".format(accuracy))
    print("f1_score on the test data is :{}".format(f1_score))
    print("mcc on the test data is :{}".format(mcc))

    return sensitivity,specifity,f1_score,mcc,accuracy


#svc = svm.SVC()
#from sklearn.model_selection import GridSearchCV
#parameters = [
#{
# 'C': [1, 3, 5, 7, 9, 11],
# 'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10],
# 'kernel': ['rbf']
#},
#{
# 'C': [1, 3, 5, 7, 9, 11],
# 'kernel': ['linear']
#}
#]
#print("search svm parameters start:")
#svm_model = GridSearchCV(svc, parameters, cv=5, n_jobs=2)
#svm_model.fit(X_encoded, y)
#print("svm_model.best_params_")
#print(svm_model.best_params_)
#print("search svm parameters done.")

#svm_model = svm.SVC(kernel='linear', C=9,probability=True)
svm_model = svm.SVC(kernel='rbf', C=5,gamma = 0.0001,probability=True)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve,auc
from scipy import interp
import matplotlib.pyplot as plt
skf = StratifiedShuffleSplit(n_splits=10, random_state = 2)
performance_list = []
tprs = []
aucs = []
fpr_mean = np.linspace(0, 1, 100)

plt.figure(figsize=(10,6))
for train_index,test_index in skf.split(X_encoded,y):
    # print("Train Index:{},Test Index:{}".format(train_index,test_index))
    X_train_encoded,X_test_encoded = X_encoded[train_index],X_encoded[test_index]
    # print(X_test_encoded)
    y_train,y_test = y[train_index],y[test_index]
    # print(y_test)
    
    # svm model training
    print("fit the model")
    #svm_model.fit(X_train_encoded, y_train)
    svm_model.fit(X_train_encoded, y_train)
    # evaluation
    X_train_score = svm_model.score(X_train_encoded, y_train)
    X_test_score = svm_model.score(X_test_encoded, y_test)
    print("training score(accuracy):{}".format(X_train_score))
    print("test score(accuracy):{}".format(X_test_score))
    y_test_pred = svm_model.predict(X_test_encoded)
    tp,tn,fp,fn = predict_comparision(y_test_pred,y_test)
    sensitivity,specifity,f1_score,mcc,accuracy = calculate_performance(tp,tn,fp,fn)
    performance_list.append([sensitivity,specifity,f1_score,mcc,accuracy])

    # prediction results (probability)
    y_test_probability = svm_model.predict_proba(X_test_encoded)
    # print(y_test_probability)
    fpr,tpr,thresholds = roc_curve(y_test,y_test_probability[:,1])
    # ROC curve for each fold
    plt.plot(fpr,tpr,linewidth=1, linestyle="-")
    # auc calculation
    m_auc = auc(fpr,tpr)
    # interp 
    tprs.append(interp(fpr_mean, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(m_auc)
    #print("fpr {}".format(fpr))
    #print("tpr {}".format(tpr))
    #print("thresholds {}".format(thresholds))
    print("=======================================")
    


#svm_model = svm.SVC(kernel='rbf', C=5, gamma= 0.0001,probability=True)
#svm_model.fit(X_train_encoded, y_train)
#X_train_score = svm_model.score(X_train_encoded, y_train)
#print(X_train_score)
#
## predict the test dataset
#
#print(y_test_hat)
#X_test_score = svm_model.score(X_test_encoded, y_test)
#print(X_test_score)
#print ("================================================================")
#
##Evaluate the performance of the trained model using the test dataset
#


#from sklearn.metrics import accuracy_score,precision_score, \
#recall_score,f1_score,cohen_kappa_score
#
#print("Sensitivity/recall:")
#print(recall_score(y_test,y_test_hat))
#
#print("precision:")
#print(precision_score(y_test,y_test_hat))
#
#print("accuracy on the test data is :")
#print(accuracy_score(y_test,y_test_hat))
#
#print("f1 score :")
#print(f1_score(y_test,y_test_hat))
#
#print("cohen_kappa_score:")   
#print(cohen_kappa_score(y_test,y_test_hat)) 

# calculate the performance for 10 fold cross validation
performance_mean = np.mean(performance_list,axis = 0)
performance_std = np.std(performance_list,axis = 0)
print("sensitivity,specifity,f1_score,mcc,accuracy")
print("performance_mean {}".format(performance_mean))
print("performance_std {}".format(performance_std))



# get the average and standard error of fpr and tpr
tpr_mean = np.mean(tprs,axis = 0)
tpr_mean[-1] = 1.0
auc_mean = np.mean(aucs)
auc_std = np.std(aucs)
plt.plot(fpr_mean,tpr_mean,linewidth=3, C = "black",linestyle="-",label = 'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc_mean,auc_std))

plt.xlim(0,1)     ## set the range of x axis
plt.ylim(0.0,1.1) ## set the range of y axis
plt.xlabel('False Postive Rate')
plt.ylabel('True Postive Rate')
plt.legend(loc = "center right")
plt.title('ROC')
plt.show()
