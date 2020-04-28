import numpy as np
import util
import matplotlib.pyplot as plt 


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    theta = clf.fit(x_train, y_train)
    yPredict = clf.predict(x_valid)
    
    # Plot decision boundary on validation set
    #x1DB = np.linspace(min(x_valid[:,0])+1.5,max(x_valid[:,0])-5) #Dataset1
    x1DB = np.linspace(min(x_valid[:,0]),max(x_valid[:,0]))   #Dataset2
    x2DB = -(theta[0][0]*x1DB + theta[1])/theta[0][1]

    
    x1Ones = []
    x2Ones = []
    x1Zeros = []
    x2Zeros = []
    #Determine which data points are  and which are 1
    for i in range(0,len(x_valid)):
        if (y_valid[i]==1):
            x1Ones.append(x_valid[i][0])
            x2Ones.append(x_valid[i][1])
        else:
            x1Zeros.append(x_valid[i][0])
            x2Zeros.append(x_valid[i][1])
    
    #Plot
    plt.scatter(x1Zeros,x2Zeros,marker='o',c="blue")  
    plt.scatter(x1Ones,x2Ones,marker='v',c="green")
    plt.plot(x1DB,x2DB,c="red")
    plt.legend(['Decision Boundary','y=0','y=1'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path,yPredict)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        nValues = len(x)
        # Initialize vectors for phi, mu_0, mu_1, sigma
        phi = 0
        mu0Num = [0, 0]
        mu0Denom = 0
        mu1Num = [0, 0]
        mu1Denom = 0
        sigma = 0

       # Compute phi, mu0, mu1
        for n in range(0, nValues):
            if (y[n] == 1):
                phi = phi + 1
                mu1Num = mu1Num + x[n]
                mu1Denom = mu1Denom + 1
            else:  # if  y==0
                mu0Num = mu0Num + x[n]
                mu0Denom = mu0Denom + 1

        phi = (1/nValues)*phi
        mu0 = mu0Num/mu0Denom
        mu1 = mu1Num/mu1Denom

        # Compute sigma
        sigma = np.zeros([2, 2])

        for n in range(0, nValues):
            diff0 = x[n]-mu0
            diff1 = x[n]-mu1

            if (y[n] == 1):
                sigma = sigma + (diff1.reshape(2, 1)*diff1)
            else:  # if  y==0
                sigma = sigma + (diff0.reshape(2, 1)*diff0)

        sigma = (1/nValues)*sigma

        # Write theta in terms of the parameters
        theta = 0.5*( np.matmul(-mu0, np.transpose(np.linalg.inv(sigma))) - \
                np.matmul(mu0, np.linalg.inv(sigma)) + \
                np.matmul(mu1, np.transpose(np.linalg.inv(sigma))) + \
                np.matmul(mu1, np.linalg.inv(sigma)) )

        theta0 = (0.5*(np.matmul((np.matmul(mu0, np.linalg.inv(sigma))), mu0) -
                       0.5*(np.matmul(np.matmul(mu1, np.linalg.inv(sigma)), mu1)))
                             + np.log((1-phi)/phi)  )

        
        theta = [theta, theta0]
        self.theta = theta
        return theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = self.theta
        yPredict = []
        for i in range(0,len(x)):
            expTerm = np.matmul(theta[0],x[i])+theta[1]
            yPredict.append(1/(1+np.exp(-expTerm)))   
        return yPredict
        # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    # main(train_path='ds2_train.csv',
    #      valid_path='ds2_valid.csv',
    #      save_path='gda_pred_2.txt')
