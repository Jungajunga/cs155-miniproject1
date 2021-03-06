
# coding: utf-8

# In[3]:

import csv 
import numpy as np
import array
import matplotlib.pyplot as plt
import random
#load scikit SVM functions
from sklearn import linear_model
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# load elctn training data
with open('/Users/danielsiebel/Desktop/(CS:CNS:EE 155) Machine Learning & Data Mining/Kaggle1/train_2008.csv','r') as dest_f: 
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
    elctn = [elctn for elctn in data_iter] 
    elctn = np.asarray(elctn)
#print(elctn.shape)     #(64668, 383)

# load elctn training data
with open('/Users/danielsiebel/Desktop/(CS:CNS:EE 155) Machine Learning & Data Mining/Kaggle1/test_2008.csv','r') as dest_fT: 
    data_iter = csv.reader(dest_fT, delimiter = ',', quotechar = '"') 
    elctn_T = [elctn_T for elctn_T in data_iter] 
    elctn_T = np.asarray(elctn_T)
#print(elctn_T.shape)     #(16001, 382)


# add bias term to data; last column = column of outputs
X1  = np.hstack((np.ones((64667,1)), elctn[1:64668,1:382].astype(np.float)))
y1  = elctn[1:64668,382].astype(np.float)
X_T  = np.hstack((np.ones((16000,1)), elctn_T[1:16001,1:382].astype(np.float)))

# let 1 = voted, -1 = did not vote
count = 0
for i in range(64667):
    if (y1[i]==2):
        y1[i]  = -1
        count += 1

#have found below columns to always have the same entry
cut_out = [1, 2, 12, 14, 16, 47, 58, 129, 130, 131, 135, 136, 137, 254, 258]
X1    = np.delete(X1, cut_out, 1) 
X_T    = np.delete(X_T, cut_out, 1) 
#print("X1.shape: ", X1.shape)    #(64667, 367)






#Compute mean and std of every column of X1,X2,X_T
Xinfo = np.zeros((2,367))
for j in range(1,367):
    Xinfo[0,j] = np.mean(X1[:,j])
    Xinfo[1,j] = np.std(X1[:,j])
    
#normalize every column of X1,X2,X_T apart from first one
for j in range(1,367):
    X1[:,j] = (X1[:,j] - Xinfo[0,j]) / Xinfo[1,j]
    X_T[:,j] = (X_T[:,j] - Xinfo[0,j]) / Xinfo[1,j]

        
clf_LogRegL2 = linear_model.LogisticRegression(penalty='l2')
clf_LogRegL2.fit(X1, y1) 
        
loss      = (1-clf_LogRegL2.score(X1, y1))*np.var(y1)
class_err = 1-np.mean((np.sign(clf_LogRegL2.predict(X1))-y1)**2)/4
        
print("   training loss: "),
print(loss),
print("   classification accuracy: "),
print(class_err)     # 77.4259


y_test2 = np.sign(clf_LogRegL2.predict(X_T))

identity = elctn_T[1:,0] # data_test is data set. this is the first column
print(identity)
y_test3 = np.array(["%.i" % w for w in y_test2.reshape(y_test2.size)]) # y_test2 is a prediction 
print(y_test3)

ListC = np.vstack((identity, y_test3))
topic = ['id', 'PES1']
results = np.vstack((topic, ListC.T))
print(results)
#with open('output.csv', 'w') as f:
#    writer = csv.writer(f)
#    writer.writerows(results)




# In[ ]:



