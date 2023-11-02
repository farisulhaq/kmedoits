from math import cos
# Imports
import pandas as pd
import numpy as np


def init_medoids(X, k):
    from numpy.random import choice
    from numpy.random import seed

    # seed(1)
    samples = choice(len(X), size=k, replace=False)
    return X[samples, :]


def compute_d_p(X, medoids, p):
    m = len(X)
    medoids_shape = medoids.shape
    if len(medoids_shape) == 1:
        medoids = medoids.reshape((1, len(medoids)))
    k = len(medoids)

    S = np.empty((m, k))

    for i in range(m):
        d_i = np.linalg.norm(X[i, :] - medoids, axis=1)
        # S[i, :] = d_i**p
        S[i, :] = d_i

    return S


def assign_labels(S):
    return np.argmin(S, axis=1)


# Full algorithm
def kmedoids(X, k, p, starting_medoids=None, max_steps=np.inf):
    medoidsAll = []
    c = []
    if starting_medoids is None:
        medoids = init_medoids(X, k)
    else:
        medoids = starting_medoids
    medoidsAll.append(medoids)
    S = np.zeros(X.shape[0])
    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids = medoids.copy()
        S_c = compute_d_p(X, medoids, p)
        c.append(S_c)
        cost = np.amin(S_c, axis=1)
        labels = assign_labels(S_c)

        if (S.sum() - cost.sum()) > 0:
            break

        S = cost

        # medoids = update_medoids(X, medoids, p)
        medoids = init_medoids(X, k)
        medoidsAll.append(medoids)
        # converged = has_converged(old_medoids, medoids)
        i += 1
    return (medoidsAll, labels, c)
