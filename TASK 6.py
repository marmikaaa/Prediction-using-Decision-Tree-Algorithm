#!/usr/bin/env python
# coding: utf-8

# # GRADUATE ROTATIONAL INTERNSHIP PROGRAM (GRIP)
# Data Science [#GRIPMARCH21]
# TASK 6: Prediction using Decision Tree Algorithm
# NAME: MARMIKA SAXENA

# In[63]:


# importing all neccassary files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:


# importing the data
iris = pd.read_csv('E:\Downloads\Iris.csv')
iris


# In[68]:


# checking the diamensions of data
iris.shape


# In[69]:


# checking the number of features in data
iris.columns


# In[70]:


# removing the inessential features
iris_1 = iris.drop(['Id'], axis= 1)

# Viewing the data
iris_1


# In[71]:


# Checking for the null values if any in given feature
print(iris_1.isnull().sum())


# In[72]:


# Plotting a pair plot for more visualization
sns.pairplot(iris_1,hue='Species')
plt.show()


# In[73]:


# Spliting the data into train and test
from sklearn.model_selection import train_test_split

# independent and dependent variable
X = iris_1.iloc[:, 0:-1]
y = iris_1.iloc[:, 4]


# In[74]:


X.head()


# In[75]:


y.head()


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[77]:


X_train.shape


# In[78]:


y_train.shape


# In[79]:


# Importing the model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Defining a variable for decision tree
iris_classifier = tree.DecisionTreeClassifier(criterion= 'entropy', random_state= 0)

# fitting the model
iris_classifier.fit(X_train, y_train)


# In[80]:


# checking the mean accuracy of the model
iris_classifier.score(X_test, y_test)


# In[81]:


# Creating a test prediction variable and testing it
tree_test_predict = pd.DataFrame({"Actual": y_test, "Predicted": iris_classifier.predict(X_test)})


# In[82]:


# Checking results
tree_test_predict.sample(n=10)


# In[83]:


# Predicting the class of a sample
# 's' is sample for which we want to predict its class
s = np.array([5.8, 3.5, 2.4, 1.50], ndmin = 2)
iris_classifier.predict(s)


# In[84]:


# Visualizing the decision tree graphically
plt.figure(figsize= (15, 15))
cols=list(X.columns.values)
tree.plot_tree(iris_classifier, filled= True, rounded= True, feature_names=cols)


# In[ ]:




