import numpy as np
import util

# Noise ~ N(0, sigma^2)
sigma = 0.5
# Dimension of x
d = 500
# Theta ~ N(0, eta^2*I)
eta = 1/np.sqrt(d)
# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

def ridge_regression(train_path, validation_path):
    """Problem 5 (d): Parsimonious double descent.
    For a specific training set, obtain theta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in scale_list.
    """
    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path) #x_shape = (200*500) y_shape = (200,)
    y_train = np.reshape(y_train, (len(y_train)),1) #now, y_shape = (200,1)
    x_valid, y_valid = util.load_dataset(validation_path)
    y_valid = np.reshape(y_valid, (len(y_valid)),1)
    N = len(y_valid)

    L_opt = sigma**2/eta**2
    L = [x * L_opt for x in scale_list] #list of lambdas
    I = np.identity(np.shape(x_train)[1])
    validation_error = []
    for lamb in L:
        A = np.matmul(np.transpose(x_train),x_train) + lamb * I
        B = np.matmul(np.transpose(x_train),y_train)
        A_inv = np.linalg.pinv(A)
        theta = np.matmul(A_inv,B)
        error = y_valid - np.matmul(x_valid,theta)
        validation_error.append(np.dot(error,error)/N)

    # *** END CODE HERE
    return validation_error

if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    val_err = np.asarray(val_err).T
    util.plot(val_err, 'doubledescent1.png', n_list)