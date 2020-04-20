import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # access the first element of the x values: print(x_train[0,1])
    # print(y_train)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path

    # Initialize values
    theta = np.zeros((2, 1))  # Initial value given in problem statement
    n = 1


    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

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
        #Initialization Values
        theta = self.theta
        alpha = self.step_size #Fixed learning rate as described in Piazza @192
        nValues = len(x)
        counter = 0;
        if (theta == None):
            theta = np.zeros([2,1])
        
        thetaPrev = np.ones([2,1])
        
        while( (np.linalg.norm(theta.T - thetaPrev.T) > self.eps) and counter < self.max_iter ): 
            #Initialize 
            theta1Vec = np.zeros([1,len(x)])
            theta2Vec = np.zeros([1,len(x)])
            #print(theta)
            for n in range(0,nValues-1):
                # Compute h_theta(x) as defined in Lecture 3 Notes
                hThetaX = 1/(1 + np.exp(np.matmul(-theta.T, x[n, 1:3])))

                # Compute the gradient of the log likelihood function
                logGradient = (y[n] - hThetaX)*x[n, 1:3]

                # Compute the Hessian of the log likelihood function wrt to theta1 and theta2 (first and second index of x vector)
                h1 = -hThetaX*(1-hThetaX)*x[n, 1]**2
                h2 = -hThetaX*(1-hThetaX)*x[n, 1]*x[n, 2]
                h4 = -hThetaX*(1-hThetaX)*x[n, 2]**2

                
                logHessian = np.array([[h1[0],h2[0]],[h2[0],h4[0]]]) 

                # Implement Newton method code as per Lecture 3 Notes
                hessInv = np.linalg.pinv(logHessian)
                
                
                theta1Vec[0][n] = (alpha*np.matmul(hessInv,logGradient))[0]
                theta2Vec[0][n] = (alpha*np.matmul(hessInv,logGradient))[1]
                
            theta1 = (-1/nValues)*np.sum(theta1Vec)
            theta2 = (-1/nValues)*np.sum(theta2Vec)
            thetaPrev = theta
            theta = theta - np.array([[theta1],[theta2]])
            print(theta)
            #theta = theta.T[0] - [theta1,theta2]
            counter = counter + 1
            #theta = np.array([theta],[theta2])
            print((np.linalg.norm(theta.T - thetaPrev.T)) ) 
            
        
     
    # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    # main(train_path='ds2_train.csv',
    #      valid_path='ds2_valid.csv',
    #      save_path='logreg_pred_2.txt')
