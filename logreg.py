import numpy as np
import util
import matplotlib.pyplot as plt 


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # access the first element of the x values: print(x_train[0,1])
    # print(y_train)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path

    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)
    y_predict = clf.predict(x_valid)
    x1DB = np.linspace(min(x_valid[:,1]), max(x_valid[:,1]))
    x2DB = -(theta[1][0]*x1DB + theta[0][0])/theta[2][0]
    #x2DB = -(theta[1][0]*x1DB + theta[0][0])/theta[2][0] #For Problem 6
    
    x1Ones = []
    x2Ones = []
    x1Zeros = []
    x2Zeros = []
    #Determine which data points are 0 and which are 1
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
    
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    #

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Initialization Values
        theta = self.theta
        nValues = len(x)
        counter = 0

        #For Problem 6d
        # rho = 1/11
        # kappa = rho/(1-rho)
        
        if (theta == None):
            theta = np.zeros([3, 1])

        thetaPrev = np.ones([3, 1])
        
    
        while((np.linalg.norm(theta.T - thetaPrev.T,1) > self.eps) and counter < self.max_iter):
            
            # Initialize
            gradientSum = [] #Size of theta-vector
            hessianSum = [] #Size of Hessian matrix
            
            for n in range(0, nValues):
                
                #For Problem 6d
                # omega = 0
                # if (y[n]==1):
                #     omega = 1/kappa
                # else:
                #     omega = 1
                
                # Compute h_theta(x) as defined in Lecture 3 Notes
                hThetaX = 1/(1 + np.exp(-np.matmul(theta.T, x[n])))

                # Compute the gradient of the log likelihood function
                logGradient = (y[n] - hThetaX)*x[n]
                # logGradient = omega*logGradient #For Problem 6d
                
                gradientSum.append(logGradient)
                # Compute the Hessian of the log likelihood function wrt to theta1 and theta2 (first and second index of x vector)
                H = np.array([[x[n, 0], x[n, 1], x[n, 2]], [
                             x[n, 1], x[n, 1]**2, x[n, 1]*x[n, 2]], [x[n, 2], x[n, 2]*x[n, 1], x[n, 2]**2]])
                
                
                logHessian = (1-hThetaX+hThetaX**2)*H
                # logHessian = omega*logHessian #For Problem 6d
            
                hessianSum.append(logHessian)
            
            #Compute the sum of the gradient and hessian
            hessian = -(1/nValues)*sum(hessianSum)
            gradient = -(1/nValues)*sum(gradientSum)
            
            #For Problem 6d
            # hessian = ( -(1+kappa)/(2*nValues) )*sum(hessianSum)
            # gradient = ( -(1+kappa)/(2*nValues) )*sum(gradientSum)
            
            #Update Theta
            thetaPrev = theta
            deltaTheta = self.step_size*(-np.linalg.solve(hessian,gradient))
            theta = theta.T - deltaTheta
            theta = theta.T
            counter = counter + 1
            # if (counter == 5):
            #     break
            print((np.linalg.norm(theta.T - thetaPrev.T,1)) )
        self.theta = theta
        return theta
            

            
    # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = self.theta
        yPredict = []
        for i in range(0,len(x)):
            yPredict.append(np.matmul(theta.T[0],x[i]))
            
        return yPredict
        # *** END CODE HERE ***


if __name__ == '__main__':
    # main(train_path='ds1_train.csv',
    #      valid_path='ds1_valid.csv',
    #      save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

