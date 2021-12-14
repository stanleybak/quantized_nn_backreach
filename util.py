"""
utilities for backreach
"""

from copy import deepcopy

import numpy as np

from star import Star

def quantize(x, delta=50):
    """round to the nearest delta (offset by delta / 2)

    for example using 50 will round anything between 0 and 50 to 25
    """

    return delta/2 + delta * round((x - delta/2) / delta)

def make_qstar(orig_star, qstate, pos_q):
    """return a subset of the star within the given quantization box"""

    # note this can likely be sped up a lot by avoiding calls to lp using sampling and testing constraints
    # or finding closest point in the box and checking that first
    # probably eventually still need lp to disprove things though

    dx, dy  = qstate[:2]

    # copy the lp and add box constraints
    star = deepcopy(orig_star)

    # dx constraints
    dims = star.a_mat.shape[1]
    zeros = np.zeros(dims)
    vec = zeros.copy()
    vec[Star.X_INT] = 1
    vec[Star.X_OWN] = -1

    star.add_dense_row(vec, dx + pos_q/2)
    star.add_dense_row(-vec, -(dx - pos_q/2))

    # dy constraints
    vec = zeros.copy()
    #vec[Star.Y_INT] = 1 # y-int is zero
    vec[Star.Y_OWN] = -1

    star.add_dense_row(vec, dy + pos_q/2)
    star.add_dense_row(-vec, -(dy - pos_q/2))

    return star
