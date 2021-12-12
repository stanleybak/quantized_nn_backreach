"""
utilities for backreach
"""

from copy import deepcopy

import numpy as np

from star import Star

def quantize(x, delta=50):
    """round to the nearest delta"""

    return delta * round(x / delta)

def make_qstar(orig_star, qstate, pos_q, vel_q):
    """return a subset of the star within the given quantization box"""

    # note this can likely be sped up a lot by avoiding calls to lp using sampling and testing constraints
    # or finding closest point in the box and checking that first
    # probably eventually still need lp to disprove things though

    dx, dy, vxo, vyo, vxi, _, _ = qstate

    # copy the lp and add box constraints
    star = deepcopy(orig_star)

    # note: constraints need to be added in the range, not the domain

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

    # vxo constraints
    vec = zeros.copy()
    vec[Star.VX_OWN] = 1

    star.add_dense_row(vec, vxo + vel_q/2)
    star.add_dense_row(-vec, -(vxo - vel_q/2))

    # vyo constraints
    vec = zeros.copy()
    vec[Star.VY_OWN] = 1

    star.add_dense_row(vec, vyo + vel_q/2)
    star.add_dense_row(-vec, -(vyo - vel_q/2))

    # vxi constraints
    vec = zeros.copy()
    vec[Star.VX_INT] = 1

    star.add_dense_row(vec, vxi + vel_q/2)
    star.add_dense_row(-vec, -(vxi - vel_q/2))

    return star
