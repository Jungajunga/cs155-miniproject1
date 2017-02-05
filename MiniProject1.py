
# coding: utf-8

# In[5]:

from sklearn import tree
import csv
import numpy as np
import matplotlib.pyplot as plt

with open ('train_2008.csv' ,'r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
    data = [data for data in data_iter] 
    data_array = np.asarray(data)
    



# In[17]:

shape = data_array.shape
identity = data_array[:,0]
y = np.array(data_array[1:,shape[1]-1],'f')
X = np.array(data_array[1:,1:381],'f') # do not consider the first row since it's identity number.


# In[24]:

# divide the training set and test set. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(y_train)


# In[26]:

clf = tree.DecisionTreeClassifier(min_samples_leaf=1000, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
clf = clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


# In[27]:

with open ('test_2008.csv' ,'r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
    data = [data for data in data_iter] 
    data_test = np.asarray(data)




# In[161]:


X_test2 = np.array(data_test[1:,1:381],'f')
Y_test2 = clf.predict(X_test2)
identity = data_test[1:,0]

        




# In[160]:

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(new)


# In[112]:




# In[ ]:



