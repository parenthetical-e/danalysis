from __future__ import division
import numpy as np


def l1(x1, x2):
    """The l1 norm loss function"""
    return np.abs(x1 - x2)


def l2(x1, x2):
    """The l2 norm loss function"""
    return np.sqrt((x1 - x2)**2)


def linf(x1, x2):
    """The l-inifinity norm loss function"""
    return np.max(np.abs(x1 - x2))


def recurrence_matrix(x1, x2, ep, lossfn=l2):
    """Create a recurrence matrix
    
    Params
    ------
    x1 : array-like, 1d
        A time series
    x2 : array-like, 1d
        Another times series (let x1 be x2 if you want the
        recurrence of a series with itself.)
    ep : numeric
        How close two points must be to e considered the same.
    lossfn : callable, lossfn(x1, x2)
        The function used to calculate the distance between
        the two time series
    """
    if x1.ndim != 1:
        raise ValueError("x1 must be a vector")
    if x2.ndim != 1:
        raise ValueError("x2 must be a vector")

    x2 = np.flipud(x2)

    i = x1.shape[0]
    j = x2.shape[0]

    mat = np.zeros((i, j), dtype=int)
    for n in range(i):
        for m in range(j):
            if lossfn(x1[n], x2[m]) <= ep:
                mat[m, n] = 1

    return mat


def correlation_d(mat):
    """
    A simple way to estimate the dimensionality of a dynamical system.

    Implementation taken from
    http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm
    """

    print("DO NOT USE. BROKEN?")

    if mat.ndim != 2:
        raise ValueError("mat must be a 2d matrix")
    if np.any(mat > 1) or np.any(mat < 0):
        raise ValueError("mat must be binary")

    N = mat.size
    g = np.diagonal(mat)
    # g = np.tril(mat, -1) # g is the sum over the heavside used in Grassberger
    # g = g[g.nonzero()]
    g = g.sum()

    return (2.0 / N * (N - 1)) * g
