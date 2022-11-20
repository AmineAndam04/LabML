import numpy as np
from decision_tree import DecisionTree

def predict(model,X):
        N, D = X.shape
        y = np.zeros(N)

        # GET VALUES FROM MODEL
        splitVariable = model._splitModel._splitVariable
        splitValue = model._splitModel._splitValue
        splitSat = model._splitModel._splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(N)

        # the case with depth=1, just a single stump.
        elif model._subModel1 is None:
            if model._splitModel._splitVariable is None:
                return model._splitModel._splitSat * np.ones(N)

            yhat = np.zeros(N)

            for m in range(N):
                if X[m, model._splitModel._splitVariable] >= model._splitModel._splitValue:
                    yhat[m] = model._splitModel._splitSat
                else:
                    yhat[m] = model._splitModel._splitNot

            y = yhat

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:,j] >= value
            splitIndex0 = X[:,j] < value
            yhat = np.zeros(N)

           
            for subModel,index in [(model._subModel0,splitIndex0),(model._subModel1,splitIndex1)]:
                N, D = X[index,].shape
                X_sub = np.round(X[index,])

                if subModel._splitModel._splitVariable is None:
                    yhat[index] = subModel._splitModel._splitSat * np.ones(N)
                    continue

                yhat_sub = np.zeros(N)

                for m in range(N):
                    if X_sub[m, subModel._splitModel._splitVariable] >= subModel._splitModel._splitValue:
                        yhat_sub[m] = subModel._splitModel._splitSat
                    else:
                        yhat_sub[m] = subModel._splitModel._splitNot
                yhat[index]=yhat_sub
            y =yhat
        return y