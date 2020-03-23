#!/usr/bin/env python
# coding: utf-8

# <h2> Support Vector Machines </h2>

# Support Vector Machine (SVM) is a supervised machine learning algorithm typically used by binary classification problems. It is a linear model for classification and regression problems. It can slove linear and non-linear problems and work well for many practical problems.
# 
# The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.SVM is an algorithm that takes the data as an input and outputs a line that separates those classes if possible.The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N - the number of features) that distinctly classifies the data points.  

# In[35]:


get_ipython().run_cell_magic('html', '', '<img src="SVM.png", width=400, height=400>')


# To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has a maximaum margin,i.e. the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence. 
# 

# <h3> Hyperplanes and Support Vectors </h3>

# In[33]:


get_ipython().run_cell_magic('html', '', '<img src="hyperplane.png", width=800, height=600>')


# A hyperplane in an n-dimensional Euclidean space is a flat, n-1 dimensional subset of that space that divides the space into two disconnected parts.For example let's assume a line to be our one dimensional Euclidean space(i.e. let's say our datasets lie on a line). Now pick a point on the line, this point divides the line into two parts. The line has 1 dimension, while the point has 0 dimension.So a point is a hyperplane of the line. 

# <h2> SVM Implementation in Python </h2>

# <b>We start by importing the libraries

# In[5]:


import numpy as np   #package for scientific computing
import matplotlib.pyplot as plt  #plotting library for 2D graphics
import pandas as pd  #library for data manipulation
import os


# <b>Importing the datasets

# In[7]:


pwd


# In[10]:


datasets = pd.read_csv(r'C:\\Users\\LENOVO\Desktop\file\Social_Network.csv')


# In[12]:


datasets.head(5)


# In[13]:


X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values


# <b>Splitting the dataset into the Training set and Testing set

# In[20]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# <b>Feature scaling

# In[21]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# <b>Fitting the classifier into the Training set 

# In[22]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)


# <b>Predicting the test set results

# In[23]:


Y_Pred = classifier.predict(X_Test)


# <b>Making the Confusion Matrix

# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)


# <b>Visualising the Test set results

# In[25]:


from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




