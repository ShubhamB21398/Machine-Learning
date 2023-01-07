# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
class linear_regression:
    
    # To initialize class default attributes only
    # Serves no other purpose
    # Invokes everytime a class object is created
    def __init__(self, learning_rate = 0.001, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    #Training the model    
    #finding values of parameters
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initializing parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent to find minima 
        # so that we have minimum loss i.e., min(MSE)
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # computing gradients
            dw = (1/n_samples)*np.dot(X.T , (y_predicted-y))
            db = (1/n_samples)*np.sum(y_predicted - y)
            
            # updating parameters
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
        
    # predicting using the parameters found
    def predict(self, X):
        y_approx = np.dot(X,self.weights)+self.bias
        return y_approx
    
    

