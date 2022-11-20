import numpy as np
from decision_stump_error import DecisionStumpErrorRate

class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self._max_depth = max_depth
        self._stump_class = stump_class
        self._subModel1 = None
        self._subModel0 = None
    

    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape
        
        # Learn a decision stump
        splitModel = self._stump_class()
        splitModel.fit(X, y)

        if self._max_depth <= 1 or splitModel._splitVariable is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump
            self._splitModel = splitModel
            return
        
        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel._splitVariable
        value = splitModel._splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:,j] >= value
        splitIndex0 = X[:,j] < value

        # Fit decision tree to each split
        self._splitModel = splitModel
        if len(splitIndex1) > 0 and len(splitIndex0) > 0 :
            self._subModel1 = DecisionTree(self._max_depth-1, stump_class=self._stump_class)
            self._subModel1.fit(X[splitIndex1], y[splitIndex1])
            self._subModel0 = DecisionTree(self._max_depth-1, stump_class=self._stump_class)
            self._subModel0.fit(X[splitIndex0], y[splitIndex0])


    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        # GET VALUES FROM MODEL
        splitVariable = self._splitModel._splitVariable
        splitValue = self._splitModel._splitValue
        splitSat = self._splitModel._splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        # the case with depth=1, just a single stump.
        elif self._subModel1 is None:
            return self._splitModel.predict(X)

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:,j] >= value
            splitIndex0 = X[:,j] < value

            y[splitIndex1] = self._subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self._subModel0.predict(X[splitIndex0])

        return y
        
