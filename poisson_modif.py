import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    print(x_train.shape)
    # print(x_train.shape) Note that the size of x_train is (251,5) whereas I want it to be (5,251)
    x_train = np.transpose(x_train)
    N = np.shape(x_train)[1]
    # this line reshapes the size of y to (1,251)
    y_train = y_train.reshape(1,N)
    init = PoissonRegression(step_size=1e-5, max_iter=1000000, eps=1e-5,
                 theta_0=np.zeros((5,1)), verbose=True, lr = 1e-5)
    theta_final = init.fit(x_train, y_train)
    print(theta_final)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    x_eval = np.transpose(x_eval)
    N_p = np.shape(x_eval)[1]
    y_eval = y_eval.reshape(1, N_p)
    pred = init.predict(x_eval)
    xx = np.zeros((1,N_p))
    xx[0,:] = [i for i in range(1,N_p+1)]
    #print(pred)

    #plt.plot(xx,y_eval,'bo',xx,pred,'ro')
    plt.plot(y_eval, pred, 'ro')
    plt.title('Question 2')
    plt.xlabel('true count ')
    plt.ylabel(' predicted expected count')
    plt.legend()
    plt.show()




    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True, lr = 1e-5):
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
        self.lr = lr

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        N = np.shape(x)[1]
        difference = [1] * 5
        iteration = 0
        e = np.zeros((1,N))
        p = np.zeros((1,N))
        eta = np.zeros((1, N))
        h = np.zeros((1, N))
        while np.max(difference) > self.eps and iteration < self.max_iter :
            iteration += 1
            print(iteration)
            eta = np.matmul(np.transpose(self.theta), x)
            for i in range(N):
                h[0, i] = eta[0, i]  # 1/(1+np.exp(-eta[0,i]))
            # find the difference, e as error with size (1,251)
            #p = np.ones((1,N)) - h
            for i in range(N):
                e[0,i] = y[0,i] - np.exp(h[0,i])
            for j in range(5):
                summation = np.dot(e[0,:], x[j,:])
                difference[j] = np.linalg.norm(self.lr*summation)
                self.theta[j,0] += self.lr*summation


        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # h shape is (1,251), N is the number of columns of h
        N_p = np.shape(x)[1]
        pred = np.zeros((1, N_p))
        pred = np.exp(np.matmul(self.theta.T, x))

        return pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
