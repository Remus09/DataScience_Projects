# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:43:44 2018

@author: RacoltaR
"""
#Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset and define variables
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the data into train/test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#Fitting simple liniar regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualizing the Training test results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test test results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()