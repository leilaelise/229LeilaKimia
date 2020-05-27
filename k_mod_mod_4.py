from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random
from matplotlib.image import imread
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    A = image
    centroids_init = []
    for i in range(num_clusters):
        rand_px = random.randint(0,len(A)-1)
        rand_py = random.randint(0, len(A)-1)
        r = A[rand_px][rand_py][0]
        g = A[rand_px][rand_py][1]
        b = A[rand_px][rand_py][2]
        centroids_init.append([r,g,b])
    # raise NotImplementedError('init_centroids function not implemented')
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    new_centroids = centroids
    A = image
    c = np.zeros((len(A),len(A)))
    dist = np.zeros((1,len(centroids)))
    count = 0
    while count < 50:
        #print(centroids)
        for x in range(len(A)):
            for y in range(len(A)):
                dist = np.zeros((1,len(centroids)))
                for j in range(len(centroids)):
                    distr = int(A[x][y][0]) - centroids[j][0]
                    distg = int(A[x][y][1]) - centroids[j][1]
                    distb = int(A[x][y][2]) - centroids[j][2]
                    dist[0][j] = distr**2 + distg**2 + distb**2
                c[x,y] = np.argmin(dist)
        for j in range(len(centroids)):
            den = 0
            numr = 0
            numg = 0
            numb = 0
            for x in range(len(A)):
                for y in range(len(A)):
                    if c[x,y] == j:
                        den += 1
                        numr += A[x][y][0]
                        numg += A[x][y][1]
                        numb += A[x][y][2]
            new_centroids[j][0] = numr/den
            new_centroids[j][1] = numg/den
            new_centroids[j][2] = numb/den
        count += 1
        centroids = new_centroids
        print(count)

    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    A = image
    c = np.zeros((len(A), len(A)))
    for x in range(len(A)):
        for y in range(len(A)):
            dist = np.zeros((1, len(centroids)))
            for j in range(len(centroids)):
                distr = int(A[x][y][0]) - centroids[j][0]
                distg = int(A[x][y][1]) - centroids[j][1]
                distb = int(A[x][y][2]) - centroids[j][2]
                dist[0][j] = distr ** 2 + distg ** 2 + distb ** 2
            j_new = np.argmin(dist)
            A[x][y][0] = centroids[j_new][0]
            A[x][y][1] = centroids[j_new][1]
            A[x][y][2] = centroids[j_new][2]
    # *** END YOUR CODE ***

    return A


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
