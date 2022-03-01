import random

import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. 
        """
        self.dataPoints = []
        self.k = k
        
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.y = y
        self.X = X
        X = X.tolist()
        y = y.tolist()
        
        for i in range(len(y)):
            
            self.dataPoints.append((y[i], X[i]))      
        
        
    def most_frequent(d):
        most_frequent = dict()
        seen = set()
        for i in range(len(d)):
            if d[i][1] in seen:
                most_frequent[d[i][1]] += 1
            else:
                most_frequent[d[i][1]] = 1
                seen.add(d[i][1])        
        return max(most_frequent, key=most_frequent.get)


                


    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        """
        
        cookie = []
        for j in range(len(X)):
            distances = []
            currentXval = X[j]
            for i in range(len(self.dataPoints)):
                
                t = self.dataPoints[i] # current datapoint key value pair
                val = t[1] # current datapoint value
                key = t[0] # current datapoint key
                
                dist = spatial.distance.euclidean(val, currentXval)
                
                distances.append((dist, t[0]))
            distances.sort(key=lambda y: y[0])
            
            most_common = KNN.most_frequent(distances[0:self.k])
            cookie.append(most_common)


        return np.array(cookie)

        
    
    def predict_numpy(self, X):

        """
        Predicts the output variable's values for the query points X using numpy (no loops).
        """
        distances = spatial.distance.cdist(X, self.X) 
        # Calculates the distances between each X and each point in self.X, 
        # returning a matrix of size len(X), len(self.X)
        
        sorted_indexes = np.argsort(distances, axis=1).transpose()[:self.k].transpose() 
        # Sorts the lowest computed distances by traversing accross distance matrix from step before
        # returning an array the size len(X), self.k
        targets = self.y[sorted_indexes] 
        # Returns the associated plant for the indexes that were sorted above 
        # returning an array of size len(X), self.k

        # Returns a 1-D array of predictions 
        # Returns array length X with the predictions for each row of X
        return stats.mode(targets,axis=1).mode.flatten() 
