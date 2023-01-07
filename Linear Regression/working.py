# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:06:49 2023

@author: shubham
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# creating a dataset
X,y = datasets.make_regression(n_samples=100, n_features= 2, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

# Class that we created manually
from Linear_regression import linear_regression

regressor = linear_regression(learning_rate = 0.01)

regressor.fit(X_train,y_train)

predicted = regressor.predict(X_test)

# To check the performance
def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)

print(mse(y_test , predicted))

# Inbuilt linear regression model
from sklearn.linear_model import LinearRegression

regressor_py = LinearRegression()
regressor_py.fit(X_train,y_train)
predicted_py = regressor_py.predict(X_test)

print(mse(y_test,predicted_py))
