"""
utilities for backreach
"""

from copy import deepcopy
import os

import numpy as np

from star import Star
from settings import Settings

from timerutil import timed, Timers

def get_tau_index(tau):
    """get the index of the network given the value of tau in seconds"""

    assert isinstance(tau, int)

    tau_list = (0, 1, 5, 10, 20, 50, 60, 80, 100)
    tau_index = -1

    if tau <= tau_list[0]:
        tau_index = 0
    elif tau >= tau_list[-1]:
        tau_index = len(tau_list) - 1
    else:
        # find the index of the closest tau value, rounding down to break ties
        
        for i, tau_min in enumerate(tau_list[:-1]):
            tau_max = tau_list[i+1]

            if tau_min <= tau <= tau_max:
                if abs(tau - tau_min) <= abs(tau - tau_max):
                    tau_index = i
                else:
                    tau_index = i+1

                break

    assert tau_index >= 0, f"tau_index not found for tau = {tau}?"
    return tau_index

def is_init_qx_qy(qx, qy):
    """is this an initial quantized location?

    returns True if any of the corners is inside the collision circle
    """

    rv = False
    pos_quantum = Settings.pos_q

    xs = (qx * pos_quantum, (qx+1) * pos_quantum)
    ys = (qy * pos_quantum, (qy+1) * pos_quantum)

    # since qstates are algigned with x == 0 and y == 0 lines,
    # we just need to check if ant of the corners are initial states
    collision_rad_sq = 500**2
    epsilon = 1e-6

    for x in xs:
        for y in ys:
            dist_sq = x*x + y*y
            
            if dist_sq < collision_rad_sq - epsilon:
                # one of corners was inside collision circle
                rv = True
                break

        if rv:
            break

    return rv

def get_num_cores():
    """get num cores available for comulation"""

    #print("DEBUG: two cores")
    #return 2
    return len(os.sched_getaffinity(0))

def to_time_str(secs):
    'return a string representation of the number of seconds'

    divisors = [1, 60, 60*60, 24*60*60, 7*24*60*60, 365*24*60*60, np.inf]
    labels = ["sec", "min", "hr", "days", "weeks", "years"]
    bounds = divisors[1:]
    digits = [3, 2, 3, 4, 4, 4]
    time_str = ""

    for divisor, digit, label, bound in zip(divisors, digits, labels, bounds):
        if secs < bound:
            time_str = f"{round(secs / divisor, digit)} {label}"
            break

    return time_str

def quantize(x, delta=50):
    """round to the nearest delta (offset by delta / 2)

    for example using 50 will round anything between 0 and 50 to 25
    """

    return delta/2 + delta * round((x - delta/2) / delta)

@timed
def make_qstar(orig_star, qstate):
    """return a subset of the star within the given quantization box"""

    pos_quantum = Settings.pos_q

    qdx, qdy = qstate[:2]
    max_x = (qdx + 1) * pos_quantum
    min_x = (qdx) * pos_quantum
    max_y = (qdy + 1) * pos_quantum
    min_y = (qdy) * pos_quantum

    # copy the lp and add box constraints
    Timers.tic('deepcopy orig_star')
    star = deepcopy(orig_star)
    Timers.toc('deepcopy orig_star')

    star.limit_dx_dy((min_x, max_x), (min_y, max_y))

    if False:
        # dx constraints
        dims = star.a_mat.shape[1]
        zeros = np.zeros(dims)
        vec = zeros.copy()
        vec[Star.X_INT] = 1
        vec[Star.X_OWN] = -1

        star.add_dense_row(vec, max_x)
        star.add_dense_row(-vec, -min_x)

        # dy constraints
        vec = zeros.copy()
        #vec[Star.Y_INT] = 1 # y-int is zero
        vec[Star.Y_OWN] = -1

        star.add_dense_row(vec, max_y)
        star.add_dense_row(-vec, -min_y)

    return star

@timed
def make_large_qstar(orig_star, qx_min, qx_max, qy_min, qy_max):
    """return a subset of the star within the given quantization box"""

    pos_quantum = Settings.pos_q

    max_x = (qx_max) * pos_quantum
    min_x = (qx_min) * pos_quantum
    max_y = (qy_max) * pos_quantum
    min_y = (qy_min) * pos_quantum

    # copy the lp and add box constraints
    Timers.tic('deepcopy orig_star')
    star = deepcopy(orig_star)
    Timers.toc('deepcopy orig_star')

    star.limit_dx_dy((min_x, max_x), (min_y, max_y))

    if False:
        # dx constraints
        dims = star.a_mat.shape[1]
        zeros = np.zeros(dims)
        vec = zeros.copy()
        vec[Star.X_INT] = 1
        vec[Star.X_OWN] = -1

        star.add_dense_row(vec, max_x)
        star.add_dense_row(-vec, -min_x)

        # dy constraints
        vec = zeros.copy()
        #vec[Star.Y_INT] = 1 # y-int is zero
        vec[Star.Y_OWN] = -1

        star.add_dense_row(vec, max_y)
        star.add_dense_row(-vec, -min_y)

    return star
