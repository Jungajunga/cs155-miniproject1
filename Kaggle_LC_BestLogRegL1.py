
# coding: utf-8

# In[4]:

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


# add bias term to data; last column = column of outputs
X1  = np.hstack((np.ones((64667,1)), elctn[1:64668,1:382].astype(np.float)))
y1  = elctn[1:64668,382].astype(np.float)

# let 1 = voted, -1 = did not vote
count = 0
for i in range(64667):
    if (y1[i]==2):
        y1[i]  = -1
        count += 1

#have found below columns to always have the same entry
cut_out = [1, 2, 12, 14, 16, 47, 58, 129, 130, 131, 135, 136, 137, 254, 258]
X1    = np.delete(X1, cut_out, 1) 
#print("X1.shape: ", X1.shape)    #(64667, 367)


#columns determined to be irrelevant by L1-regularization
zero_indices = [  1,   2,   3,   7,  10,  11,  12,  13,  14,  15,  17,  18,  22,  23,  24,   27,  28,  29,  30,  31,  35,  38,  41,  44,  45,  47,  48,  52,  53,  54,  59,  60,  61, 
  62,  63,  64,  66,  67,  69,  70,  71,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82, 
  83,  85,  86, \
  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, \
 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, \
 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, \
 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, \
 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, \
 178, 180, 181, 182, 183, 184, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, \
 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, \
 217, 218, 219, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 231, 232, 234, 235, 236, \
 237, 238, 239, 240, 242, 243, 244, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, \
 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, \
 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, \
 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, \
 311, 312, 313, 314, 315, 316, 318, 320, 325, 326, 327, 328, 329, 330, 331, 332, 335, 336, \
 337, 338, 340, 341, 342, 343, 344, 345, 346, 347, 351, 352, 353, 354, 355, 357, 360, 361, \
 362, 363, 364, 365, 366]

X1    = np.delete(X1, zero_indices, 1) 
#print("X1.shape: ", X1.shape)    #(64667, 56)





#Compute mean and std of every column of X1,X2,X_T
Xinfo = np.zeros((2,56))
for j in range(1,56):
    Xinfo[0,j] = np.mean(X1[:,j])
    Xinfo[1,j] = np.std(X1[:,j])
    
#normalize every column of X1,X2,X_T apart from first one
for j in range(1,56):
    X1[:,j] = (X1[:,j] - Xinfo[0,j]) / Xinfo[1,j]

        
clf_LogRegL1 = linear_model.LogisticRegression(penalty='l1')
clf_LogRegL1.fit(X1, y1) 
        
loss      = (1-clf_LogRegL1.score(X1, y1))*np.var(y1)
class_err = 1-np.mean((np.sign(clf_LogRegL1.predict(X1))-y1)**2)/4
        
print("   training loss: "),
print(loss),
print("   classification accuracy: "),
print(class_err)     # 77.4259


y_test2 = np.sign(clf_Lasso.predict(X_T))

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



