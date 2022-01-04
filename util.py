"""
utilities for backreach
"""

from copy import deepcopy

import numpy as np

from star import Star
from settings import pos_quantum

from timerutil import timed

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

    qdx, qdy = qstate[:2]
    max_x = (qdx + 1) * pos_quantum
    min_x = (qdx) * pos_quantum
    max_y = (qdy + 1) * pos_quantum
    min_y = (qdy) * pos_quantum

    # copy the lp and add box constraints
    star = deepcopy(orig_star)

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
