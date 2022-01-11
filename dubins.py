"""
quantized backreach code for dynamics
"""

from typing import Tuple, List
from math import sin, cos, pi

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from star import Star
from settings import Settings

def init_to_constraints(qx: int, qy: int,
                        qv_own_min: int, qv_int_min: int, qtheta1_min: int):
    """convert initial variables to box bounds and constraints in linear space

    returns box, a_mat, b_vec with Ax <= b constraints
    """

    box: List[Tuple[float, float]] = []
    a_mat: List[List[float]] = []
    b_vec: List[float] = []

    pos_quantum = Settings.pos_q
    vel_quantum = Settings.vel_q
    theta1_quantum = Settings.theta1_q

    # convert to float ranges for box
    x = qx * pos_quantum, (qx + 1) * pos_quantum
    y = qy * pos_quantum, (qy + 1) * pos_quantum

    qv_own = qv_own_min, qv_own_min + 1
    qv_int = qv_int_min, qv_int_min + 1
    qtheta1 = qtheta1_min, qtheta1_min + 1

    if vel_quantum != 0:
        v_own = (qv_own[0] * vel_quantum, qv_own[1] * vel_quantum)
        v_int = (qv_int[0] * vel_quantum, qv_int[1] * vel_quantum)
    else:
        assert qv_own[0] + 1 == qv_own[1] and qv_int[0] + 1 == qv_int[1]
        # fixed velocities
        
        v_own = qv_own[0], qv_own[0]
        v_int = qv_int[0], qv_int[0]
        
    theta1 = (qtheta1[0] * theta1_quantum, qtheta1[1] * theta1_quantum)

    box.append(x)
    box.append(y)

    # compute vx and vy from theta1 and v_own
    tol = 1e-9

    # make sure limit will be one of the four sampled combinations
    assert theta1[0] < theta1[1] and theta1[0] + tol >= 0 and theta1[1] - tol <= 2*pi
    assert not (theta1[0] + tol < pi/2 < theta1[1] - tol)
    assert not (theta1[0] + tol < pi < theta1[1] - tol)
    assert not (theta1[0] + tol < 3*pi/2 < theta1[1] - tol)
    
    vx = []
    vy = []

    for theta1_scalar in theta1:
        for v_own_scalar in v_own:
            vx.append(cos(theta1_scalar) * v_own_scalar)
            vy.append(sin(theta1_scalar) * v_own_scalar)

    box.append((min(vx), max(vx)))
    box.append((min(vy), max(vy)))

    # make pie constraints

    # top-left
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = -sin(theta1[1])
    row[Star.VY_OWN] = cos(theta1[1])
    a_mat.append(row)
    b_vec.append(0)

    # bottom-right
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = sin(theta1[0])
    row[Star.VY_OWN] = -cos(theta1[0])
    a_mat.append(row)
    b_vec.append(0)

    # bottom-left
    p1 = np.array([cos(theta1[1]), sin(theta1[1])]) * v_own[0]
    p2 = np.array([cos(theta1[0]), sin(theta1[0])]) * v_own[0]
    delta = p1 - p2
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = -delta[1]
    row[Star.VY_OWN] = delta[0]
    r = np.array([-delta[1], delta[0]]).dot(p1)
    a_mat.append(row)
    b_vec.append(r)

    # top right (#1, left one)
    p1 = np.array([cos(theta1[1]), sin(theta1[1])])
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = p1[0]
    row[Star.VY_OWN] = p1[1]
    a_mat.append(row)
    b_vec.append(v_own[1])

    # top right (#2, right one)
    p1 = np.array([cos(theta1[0]), sin(theta1[0])])
    row = [0.0] * Star.NUM_VARS
    row[Star.VX_OWN] = p1[0]
    row[Star.VY_OWN] = p1[1]
    a_mat.append(row)
    b_vec.append(v_own[1])

    ############
    box.append((0.0, 0.0)) # x_int
    box.append(v_int) # vx_int

    return box, a_mat, b_vec

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
