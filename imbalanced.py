import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt 
#sys.path.append('../linearclass')
sys.path[0] = '/Users/leila/Desktop/ps1/src/linearclass'

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression
sys.path[0] = '/Users/leila/Desktop/ps1/src/imbalanced'
# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')
    

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)  
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)
    #print(theta)
    y_predict = clf.predict(x_valid)
    np.savetxt(output_path_upsampling,y_predict,delimiter=',')
    
    #Compute A, A_balanced, A_0, A_1
    
    #Compute accuracy 
    #Initialize true positive, neg, false pos, neg counts
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0

    #Loop through predicted probabilites and compute the #s of TP,FP,TN,FN
    for n in range(0,len(y_predict)):
        if (y_predict[n]>0.5 and y_valid[n]==1): #True Positive
            truePos = truePos + 1
        elif (y_predict[n]>0.5 and y_valid[n]==0): #False Positive
            falsePos = falsePos + 1
        elif (y_predict[n]<0.5 and y_valid[n]==0): #True Negative
            trueNeg = trueNeg + 1
        elif (y_predict[n]<0.5 and y_valid[n]==1): #False Negative
            falseNeg = falseNeg + 1
    
    accuracy = (truePos + trueNeg)/len(y_predict)
    posAccuracy = truePos/(truePos + falseNeg)
    negAccuracy = trueNeg/(trueNeg + falsePos)
    balancedAccuracy =  0.5*(posAccuracy + negAccuracy)
    
    print(accuracy)
    print(posAccuracy)
    print(negAccuracy)
    print(balancedAccuracy)
    
    x1DB = np.linspace(min(x_valid[:,1]),max(x_valid[:,1])) 
    x2DB = -(theta[1][0]*x1DB + theta[0][0])/theta[2][0]
    
    x1Ones = []
    x2Ones = []
    x1Zeros = []
    x2Zeros = []
    #Determine which data points are  and which are 1
    for i in range(0,len(x_valid)):
        if (y_valid[i]==1):
            x1Ones.append(x_valid[i][1])
            x2Ones.append(x_valid[i][2])
        else:
            x1Zeros.append(x_valid[i][1])
            x2Zeros.append(x_valid[i][2])
    
    #Plot
    plt.scatter(x1Zeros,x2Zeros,marker='o',c="blue")  
    plt.scatter(x1Ones,x2Ones,marker='v',c="green")
    plt.plot(x1DB,x2DB,c="red")
    plt.legend(['Decision Boundary','y=0','y=1'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    

    
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
