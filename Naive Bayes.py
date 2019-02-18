#!/usr/bin/env python
# coding: utf-8

# In[6]:





# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Create a Gaussian Classifier
model = GaussianNB()





# In[7]:


# Train DGaussian Classifier
model = model.fit(X_train,y_train)


# In[8]:


#Predict the response for test dataset
y_pred = model.predict(X_train)

# Predict and print the label for the new data point X_new
new_prediction = model.predict(X_train)
print("Prediction: {}".format(new_prediction))


# In[10]:





# In[ ]:




