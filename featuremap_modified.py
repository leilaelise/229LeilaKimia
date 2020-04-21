import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        N = len(X)
        # X has size of (N,1)
        new_x = np.zeros((4,N))
        for i in range(4):
            new_x[i,:] = np.power(X.T,i)
        LHS = np.zeros((4,4))
        RHS = np.zeros((4,1))
        theta_trans = np.zeros((4,1))
        LHS = np.matmul(new_x,new_x.T)
        RHS = np.matmul(new_x,y.T)
        theta_trans = np.linalg.solve(LHS, RHS)
        theta = theta_trans.T

        return theta
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        N = len(X)
        # X has size of (N,1)
        new_x = np.zeros((k+1,N))
        for i in range(k+1):
            new_x[i,:] = np.power(X.T,i)

        return new_x
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y_o=util.load_dataset(train_path,add_intercept=False)
    plot_x = np.ones([1000, 1])
    plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    N = len(train_x)
    train_y = np.reshape(train_y_o,(1,N))
    #plt.figure()
    plt.scatter(train_x[:, 0], train_y_o)
    #plt.show()
    color = ['b','y','g','m','c','r']
    c= 0 #this keeps tack of the color
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        init = LinearModel()
        new_x = init.create_poly(k, train_x)
        x_valid = init.create_poly(k, plot_x)
        xx = np.linspace(-factor*np.pi, factor*np.pi, 1000)
        xx = np.reshape(xx, (1, 1000))

        LHS = np.zeros((k+1,k+1))
        RHS = np.zeros((k+1,1))
        theta_trans = np.zeros((k+1,1))
        LHS = np.matmul(new_x,new_x.T)
        RHS = np.matmul(new_x,train_y.T)
        theta_trans = np.linalg.solve(LHS, RHS)
        theta = theta_trans.T

        plot_y = np.matmul(theta, x_valid)
        #print(plot_y)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(xx, plot_y,'.',color = color[c])#, label='k=%d' % k)
        c += 1
        #plt.show()
    #plt.legend()
    plt.show()
    #plt.legend()
    #plt.savefig(filename)
    #plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    '''# *** START CODE HERE FOR PART B***
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    # the original x data, without intercept is (70*1), I don't change it here
    # the original y data has 70 rows, I want it to have size of (1,70) and I change it here
    N = len(train_x)
    train_y = np.reshape(train_y,(1,N))

    init = LinearModel()
    theta = init.fit(train_x, train_y)
    #print(theta)

    # check
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    N = len(x_eval)
    y_eval = np.reshape(y_eval,(1,N))
    # X has size of (N,1)
    new_x = np.zeros((4, N))
    for i in range(4):
        new_x[i, :] = np.power(x_eval.T, i)
    pred = np.matmul(theta,new_x)
    xx = np.zeros((1, N))
    xx[0, :] = [i for i in range(1, N + 1)]
    plt.plot(xx,y_eval,'bo',xx,pred,'ro')
    plt.show()
    # *** END CODE HERE ***'''

    # *** START CODE HERE FOR PART C***
    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
