import numpy as np
import utils
from decision_stump_error import DecisionStumpErrorRate


    
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?

class DecisionStumpInfoGain(DecisionStumpErrorRate):
    def __init__(self, loss=utils.loss_l0):
        super().__init__(loss)
        self._InfoGain=None
    
    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        if len(count)>0 :
            y_mode = np.argmax(count) 
        else :
            return 

        self._splitSat = y_mode
    

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return
        
        p = count/np.sum(count)
        parent_entropy = entropy(p)
        self._InfoGain = 0

        self._minError = np.sum(y != y_mode)/N

        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] >= value])
                y_not = utils.mode(y[X[:,d] < value])

                if y_sat == y_not:
                    y_not= (y_not+ 1)%2

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] < value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)/N

                # Info Gain:
                count_sat = np.bincount(y[X[:,d] >= value])
                count_not = np.bincount(y[X[:,d] < value])
                entropy_sat= entropy(count_sat/np.sum(count_sat))
                entropy_not= entropy(count_not/np.sum(count_not))
                entropy_global= (np.sum(count_sat)/N)*entropy_sat + (np.sum(count_not/N))*entropy_not
                InfoGain = parent_entropy - entropy_global

                # Compare to minimum error so far
                if InfoGain > self._InfoGain:
                    # This is the lowest error, store this value
                    self._minError = errors
                    self._splitVariable = d
                    self._splitValue = value
                    self._splitSat = y_sat
                    self._splitNot = y_not
                    self._InfoGain = InfoGain 
    


    
"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
