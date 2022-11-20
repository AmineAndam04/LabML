# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              
import pandas as pd                             
import matplotlib.pyplot as plt                 
from scipy.optimize import approx_fprime        
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`


""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

import utils
from decision_stump import DecisionStumpEquality
from decision_stump_generic import DecisionStumpEqualityGeneric
from decision_stump_error import DecisionStumpErrorRate
from decision_stump_info import DecisionStumpInfoGain
from decision_tree import DecisionTree



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        leafs=np.log2(64)
        nodes=int(np.log2(65)-1)
        print("The minimum depth of a binary tree with 64 leaf nodes is:",leafs)
        print("The minimum depth of binary tree with 64 nodes (includes leaves and all other nodes) is:",nodes) 
    
    elif question == "1.2":
        # YOUR ANSWER HERE
        print("The running time of the function", "func1 ", "is: O(N) ")
        print("The running time of the function", "func2 ", "is: it depends on the algorithm we used to compute np.zeros()  ")
        print("The running time of the function", "func3 ", "is: O(1) ")
        print("The running time of the function", "func4 ", "is: O(N^2)  ")
    
    elif question == "2.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values
        statistics=df.describe([.05,.25,.50,.75,.95])
        statistics.loc["median"]=df.median()
        mode_values=df.mode()
        print("minimum, maximum, mean, median :")
        print(statistics.loc[["min","max","mean","median"]])
        conti=input('Do you want to continue (Show the mode) [y/n]')
        if conti in ["y",'Y',1] :
            print("Mode :For some variables we find lots of possible values of mode")
            print(mode_values)
            print("\nClearlythe mode is not a reliable estimate of the most “common” value because we are talking about a continous variables")
            print("An alternative is to plot the  probability distribution or the histogram and look for any value x at which its density function has a locally maximum value")
            conti=input('Do you want to continue (Show the quantiles) [y/n]')
            if conti in ["y",'Y',1] :
                print("The 5%, 25%, 50%, 75%, and 95% quantiles of all values")
                print(statistics.loc[["5%", "25%", "50%", "75%", "95%"]])
                conti=input('Do you want to continue (Show the highest and lowest means) [y/n]')
                if conti in ["y",'Y',1] :
                    print("The region with the highest means")
                    print(dict(statistics.loc["mean"][statistics.loc["mean"]==statistics.loc["mean"].max()]))
                    print("The region with the lowest means")
                    print(dict(statistics.loc["mean"][statistics.loc["mean"]==statistics.loc["mean"].min()]))
                    conti=input('Do you want to continue (Show the highest and lowest variances) [y/n]')
                    if conti in ["y",'Y',1] :
                        print("The highest variances")
                        print(dict(statistics.loc["std"][statistics.loc["std"]==statistics.loc["std"].max()]))
                        print("The lowest variances")
                        print(dict(statistics.loc["std"][statistics.loc["std"]==statistics.loc["std"].min()]))

       
    elif question == "2.2":
        
        # YOUR CODE HERE : modify HERE
        figure_dic = {'A':4,
                      'B':3,
                      'C':2,
                      'D':1,
                      'E':6,
                      'F':5}
        expl_dic = {'A':" it is showing the illness percentages over time (time series)",
                      'B':"it is a boxplot for each week, showing the distribution across regions",
                      'C':"it is showing the distribution of each the values in our data",
                      'D':" we have a histogram per variable",
                      'E':"the correlation in plot F is stronger",
                      'F':"we see a stong linear correlation, we can draw a straight line"}
        for label in "ABCDEF":
            print("Match the plot", label, "with the description number: ",figure_dic[label],". Because",expl_dic[label])
        
    elif question == "3.1":
        # 1: Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3: Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with Equality rule error: %.3f"
              % error)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_1_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        print("")
        # YOUR ANSWER HERE
        print("Question: It makes sense to use an  equality-based splitting rule rather than the threshold-based splits when ???" )
        print("Answer: No it does not make any sense.It is not a realistic approach")
        print("we can see from the plot that the decision boundary of equality-based model is just a line which is far from reality")

        print("\nQuestion: Is there a particular type of features for which it makes sense to use an equality-based splitting rule ratherthan the threshold-based splits we discussed in class?")
        print("Answer: Yes, for binary variable (it only takes two values) ")
    elif question == "3.2":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the generic decision stump
        model = DecisionStumpEqualityGeneric()
        
        y_pred = model.fit_predict(X,y)

        error = model.score(X, y)
        print("Decision Stump Generic rule error: %.3f" % error)
        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_2_decisionBoundaryGeneric.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "3.3":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the inequality decision stump
        model = DecisionStumpErrorRate()
        
        y_pred = model.fit_predict(X,y)

        error = model.score(X, y)

        print("Decision Stump with inequality rule error: %.3f" % error)

        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_decisionBoundaryInequality.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
               
    elif question == "3.4":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the decision stump with info gain
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)

        print("Decision Stump with info gain rule error: %.3f" % error)

        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_decisionBoundaryInfoGain.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    
    elif question == "3.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)
        import simple_decision as sd
        y_pred = model.predict(X)
        y_pred2= sd.predict(model,X)
        error = np.mean(y_pred != y)
        error2 = np.mean(y_pred2 != y)
        
        print("Error using predict function of DecisionTree class: %.3f" % error)
        print("Error using hard-coded version of the predict function: %.3f" % error2)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q3_5_decisionBoundaryDecisionTree.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        

    elif question == "3.6":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]

        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try
       
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="errorrate")
        
        
        t = time.time()
        my_tree_errors_infogain = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            
            model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        print("\nFor the three models, the training error is decreasing as we increase the depth, but not with the same speed, and they don't reach the same values.")
        print("\nIn fact, with depth equal to zero, the gap between the models is very narrow (except for errorrate model). As we increase the depth, the gap becomes more significant.")
        print("\nThe error rate of the sklearn model decreases until it equals zero, which may suggest that the model is overfitting. Because the training error of the error rate model stops decreasing between depths 4 and 5, we do not need to train our model with depths greater than 5.")
        print("\nThe infogain model is better than the errorrate model, it has less training error starting at depth 6, but it is still worse than sklearn model.")
        print("These results are not surprising given that sklearn is a library built specifically to train ML models and uses more sophisticated tools.")
        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_6_tree_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=1)
        model.fit(X,y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nBoundary saved as '%s'" % fname)

    elif question == "3.7":
        print("Question: In the previous section you compared different implementations of a machine learning algorithm. Let’s say that two approaches produce the exact same curve of classification error rate vs. tree depth. Does this conclusively demonstrate that the two implementations are the same? If so, why? If not, what other experiment might you perform to build confidence that the implementations are probably equivalent?")
        print("\nAnswear : We cannot say that they are exactly the same implementation because there are some factors we should consider, such as implementation time, especially if we are working with large scale projects. Moreover, the curve of classification error rate vs. tree depth is built using specific training data, perhaps if we used another dataset, we would get a different curve. We cannot make a general judgment from just one example of training data.")


    else:
        print("No code to run for question", question)