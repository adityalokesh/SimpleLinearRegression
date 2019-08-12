#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 05:45:37 2019

@author: chrx
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
data=pd.read_csv('/home/chrx/Downloads/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
# the above line is the path of the CSV folder place the path of the file

#Splitting the data as the dependent and the independent variable data 
X=data.iloc[:, 0:1].values
y=data.iloc[:, 1:2].values

# calling the test_train _split model to tell the model how much data is train and how much is test data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# importing the linear regression model to create a regression object

from sklearn.linear_model import LinearRegression
linear_obj=LinearRegression()

# training the model on data split for training
linear_obj.fit(X_train,y_train)

#making prediction on the same trained data to see how  well trained
y_pred=linear_obj.predict(X_train)

#plotting the real values and the best fit line for the same X_train data 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,y_pred,color='blue')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression-red is actual data,blue line is best fit line for the actual data')
plt.show()

#very simple easy to read linear regressor
