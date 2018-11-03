# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:56:50 2018

@author: RacoltaR
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting the data into training/test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#1 Build linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
y_linReg = regressor.predict(x)

#2 Build polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
y_PolyReg = lin_reg2.predict(x_poly)

#Visualizing the results
plt.scatter(x,y, color = 'red')
plt.plot(x,y_linReg, color = 'blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.plot(x, y_PolyReg, color = 'yellow')
plt.show()