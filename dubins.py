"""
quantized backreach code for dynamics
"""

from typing import Tuple, List
from math import sin, cos, pi

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from star import Star

def init_to_constraints(v_own: Tuple[float, float], v_int: Tuple[float, float],
                 x: Tuple[float, float], y: Tuple[float, float], psi: Tuple[float, float]):
    """convert initial variables to box bounds and constraints in linear space

    returns box, a_mat, b_vec with Ax <= b constraints
    """

    rv: List[Tuple[float, float]] = []
    a_mat: List[List[float]] = []
    b_vec: List[float] = []

    rv.append(x)
    rv.append(y)

    # compute vx and vy from psi and v_own
    tol = 1e-9

    # make sure limit will be one of the four sampled combinations
    assert psi[0] < psi[1] and psi[0] + tol >= 0 and psi[1] - tol <= 2*pi
    assert not (psi[0] + tol < pi/2 < psi[1] - tol)
    assert not (psi[0] + tol < pi < psi[1] - tol)
    assert not (psi[0] + tol < 3*pi/2 < psi[1] - tol)
    
    vx = []
    vy = []

    for psi_scalar in psi:
        for v_own_scalar in v_own:
            vx.append(cos(psi_scalar) * v_own_scalar)
            vy.append(sin(psi_scalar) * v_own_scalar)

    rv.append((min(vx), max(vx)))
    rv.append((min(vy), max(vy)))

    # make pie constraints

    # top-left
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = -sin(psi[1])
    row[Star.VY_OWN] = cos(psi[1])
    a_mat.append(row)
    b_vec.append(0)

    # bottom-right
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = sin(psi[0])
    row[Star.VY_OWN] = -cos(psi[0])
    a_mat.append(row)
    b_vec.append(0)

    # bottom-left
    p1 = np.array([cos(psi[1]), sin(psi[1])]) * v_own[0]
    p2 = np.array([cos(psi[0]), sin(psi[0])]) * v_own[0]
    delta = p1 - p2
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = -delta[1]
    row[Star.VY_OWN] = delta[0]
    r = np.array([-delta[1], delta[0]]).dot(p1)
    a_mat.append(row)
    b_vec.append(r)

    # top right (#1, left one)
    p1 = np.array([cos(psi[1]), sin(psi[1])])
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = p1[0]
    row[Star.VY_OWN] = p1[1]
    a_mat.append(row)
    b_vec.append(v_own[1])

    # top right (#2, right one)
    p1 = np.array([cos(psi[0]), sin(psi[0])])
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = p1[0]
    row[Star.VY_OWN] = p1[1]
    a_mat.append(row)
    b_vec.append(v_own[1])

    ############
    rv.append((0.0, 0.0)) # x_int
    rv.append(v_int) # vx_int

    return rv, a_mat, b_vec

@lru_cache(maxsize=None)
def get_time_elapse_mat(command1, dt):
    '''get the matrix exponential for the given command

    state: x_own, y_own, vx_own, vy_own, x_int, vx_int
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    
    dtheta1 = (y1 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 1], # x_int' = vx_int
        [0, 0, 0, 0, 0, 0] # vx_int' = 0
        ], dtype=float)

    assert a_mat.shape[0] == a_mat.shape[1]

    return expm(a_mat * dt)
