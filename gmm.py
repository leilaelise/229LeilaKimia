import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal
import random 
 
PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def LogLikelihoodSemi(m,dim,sigma,x,x_tilde,z_tilde,mu,phi,alpha):
    k = 4
    sumX = 0
    n = len(x_tilde[:,0])
    for i in range(0,m):
        sumZ = 0
        for j in range(0,k):
            sumZ = sumZ + pXZ_pZ(dim,sigma[j],x[i],mu[j],phi[j])
        sumX = sumX + np.log(sumZ)
    
    sumSupervised = 0
    for i in range(0,n):
        idx = int(z_tilde[i])
        sumSupervised = sumSupervised + np.log(pXZ_pZ(dim,sigma[idx],x_tilde[i],mu[idx],phi[idx]))
                                            
    totalSum = sumX[0] + alpha*sumSupervised
    
    return totalSum

def LogLikelihood(m,dim,sigma,x,mu,phi):
    k = 4
    sumX = 0
    for i in range(0,m):
        sumZ = 0
        for j in range(0,k):
            sumZ = sumZ + pXZ_pZ(dim,sigma[j],x[i],mu[j],phi[j])
        sumX = sumX + np.log(sumZ)
    return sumX[0]

def pXZ_pZ(dim,sigma,x,mu,phi):
    #Gaussian 
    eta = 1/( ( (2*np.pi)**(dim/2) )*np.linalg.det(sigma)**(1/2) )
    bbT = np.matmul( np.matmul((x - mu),np.linalg.pinv(sigma)), (x-mu))
    pXZ = eta*np.exp(-0.5*bbT)
    return pXZ*phi 
   
    
#Compute the covariance matrix of samples
def covMatrix(dim,x,mu,start,stop):

    #Initialize
    sigma = np.zeros([dim, dim])

    for n in range(start,stop):
        diff0 = x[n]-mu
        sigma = sigma + (diff0.reshape(2, 1)*diff0)

    sigma = (1/(stop-start))*sigma
    return sigma

def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    k = 4
    dim = 2
    m = len(x[:,0])
    mu = np.zeros([k,dim])
    sigma = []
    #Initialize mu and sigma into k=4 groups
    #Algorithm: Randomly pick four mu from divided data and compute covariance according to that mu
    #random.shuffle(x)
    for i in range(0,k):
        start = int(i*m/k)
        stop = int((i+1)*m/k)
        idx =np.random.randint(start,stop)
        mu[i] = x[idx]
        vec = x[start:stop]
        sigma.append(covMatrix(dim,x,mu[i],start,stop))  
    
    
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.zeros([k,1])
    for i in range(0,k):
        phi[i] = 1/k
          
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.zeros([m,k])
    for i in range(0,m):
        w[i] = 1/k
    
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    #Dimensions
    m = len(x[:,0])
    k = 4
    dim = 2
    
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    prev_ll = 0
    ll  = LogLikelihood(m,dim,sigma,x,mu,phi)
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        prev_ll = ll
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE

        # (1) E-step: Update your estimates in w
        for i in range(0,m): 
            for j in range(0,k):                          
                denom = 0
                num = 0
                pZX = pXZ_pZ(dim,sigma[j],x[i],mu[j],phi[j])
                w[i][j] = pZX     
            w[i] = w[i]/sum(w[i])    

               
        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        #Update phi
        for j in range(0,k):
            sumPhi = 0
            for i in range(0,m):
                sumPhi = sumPhi + w[i][j]
            phi[j] = sumPhi
        phi = (1/m)*phi

        #Update mu
        for j in range(0,k):
            numSum = 0
            denomSum = 0
            for i in range(0,m):
                numSum = numSum + w[i][j]*x[i]
                denomSum = denomSum + w[i][j]
            mu[j] = numSum/denomSum 
        

        #Update sigma
        for j in range(0,k):
            sigmaNum = 0
            sigmaDenom = 0
            for i in range(0,m):
                sigmaNum = sigmaNum + w[i][j]*( (x[i]-mu[j])*(x[i]-mu[j]).reshape(dim,1) )
                sigmaDenom = sigmaDenom + w[i][j]          
            sigma[j] = sigmaNum/sigmaDenom
                            
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        prev_ll = ll
        ll = LogLikelihoodSemi(m,dim,sigma,x,mu,phi)
        
        print( abs(ll - prev_ll) )
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    #Dimensions
        #Dimensions
    m = len(x[:,0])
    n = len([x_tilde[:,0]])
    k = 4
    dim = 2
    
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    prev_ll = 0
    ll = LogLikelihoodSemi(m,dim,sigma,x,x_tilde,z_tilde,mu,phi,alpha)
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for i in range(0,m): 
            for j in range(0,k):                          
                denom = 0
                num = 0
                pZX = pXZ_pZ(dim,sigma[j],x[i],mu[j],phi[j])
                w[i][j] = pZX     
            w[i] = w[i]/sum(w[i])    
        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        #Update phi
        for j in range(0,k):
            sumPhi = 0
            for i in range(0,m):
                sumPhi = sumPhi + w[i][j]
            phi[j] = sumPhi + alpha*n
        phi = (1/(m+alpha*n))*phi

        #Update mu
        for j in range(0,k):
            numSum = 0
            numTilde = 0
            denomSum = 0

            for i in range(0,m):
                numSum = numSum + w[i][j]*x[i]
                denomSum = denomSum + w[i][j]
                   
            for i in range(0,n):
                numTilde = numTilde + x_tilde[i]
                
            mu[j] = (numSum + alpha*numTilde)/(denomSum + alpha*n)
        

        #Update sigma
        for j in range(0,k):
            sigmaNum = 0
            sigmaDenom = 0
            tildeSum = 0
            for i in range(0,m):
                sigmaNum = sigmaNum + w[i][j]*( (x[i]-mu[j])*(x[i]-mu[j]).reshape(dim,1) )
                sigmaDenom = sigmaDenom + w[i][j]  
            
            for i in range(0,n):
                tildeSum = tildeSum + ( (x_tilde[i]-mu[j])*(x_tilde[i]-mu[j]).reshape(dim,1) )
                
            sigma[j] = (sigmaNum+tildeSum)/(sigmaDenom +alpha*n)       
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = LogLikelihoodSemi(m,dim,sigma,x,x_tilde,z_tilde,mu,phi,alpha)
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***
        print( abs(ll - prev_ll) )
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
       # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
         main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
