import numpy as np
from numpy.core.fromnumeric import mean 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, task=1):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.task = task
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        if isinstance(X, np.ndarray):
            X_array = X
        else:
            X_array = X.to_numpy()
        m_samples, n_features = X_array.shape
        # Normalize data
        for i in range(n_features):
            mean = np.mean(X_array[:,i])
            std = np.std(X_array[:,i])
            X_array[:,i] = (X_array[:,i] - mean) / (std)

        if self.task == 2:
            # Transfer to polar coordinates because of circular data
            R = np.square(X_array[:,0]) + np.square(X_array[:,1])
            # Using the angle as a feature reduced the accuracy 
            # due to overfitting caused by the data being circular and not elliptical
            # Angle = np.arctan(X_array[:,1]/X_array[:,0]) 

            X_array = R
            n_features = 1
            # X_array[:,1] = Angle

        lr = 0.03
        theta = np.random.uniform(-10,10, n_features)
        bias = 1
        
        n_iterations = 10000
        for i in range(n_iterations):
            if self.task == 2:
                z = theta * X_array + bias
            else:
                z = np.dot(theta, X_array.T) + bias
            y_pred = sigmoid(z)
            grad_theta = np.dot((y-y_pred).T, X_array)/m_samples
            theta = theta + lr*grad_theta
            bias = bias + lr*np.mean(y-y_pred)
        self.bias = bias
        self.theta = theta
        

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if isinstance(X, np.ndarray):
            X_array = X
        else:
            X_array = X.to_numpy()
        n_features = X_array.shape[1]
        for i in range(n_features):
            mean = np.mean(X_array[:,i])
            std = np.std(X_array[:,i])
            X_array[:,i] = (X_array[:,i] - mean) / (std)
        if self.task == 2:
            # Transfer to polar coordinates because of circular data
            R = np.square(X_array[:,0]) + np.square(X_array[:,1])
            # Angle = np.arctan(X_array[:,1]/X_array[:,0])

            X_array = R
            # X_array[:,1] = Angle
            return sigmoid(self.theta * X_array + self.bias)

        return sigmoid(np.dot(self.theta, X_array.T) + self.bias)
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        
