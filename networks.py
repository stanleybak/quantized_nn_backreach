"""
Neural network interface for quantized backreach

Stanley Bak, 12-2021
"""

from functools import lru_cache

import os
from math import pi, atan2

import numpy as np

import onnxruntime as ort

def qstate_cmd(alpha_prev, qstate, stdout=False):
    """get the command at the given quantized state"""

    assert isinstance(alpha_prev, int) and 0 <= alpha_prev <= 4, f"alpha_prev was {alpha_prev}"

    if stdout:
        print(f"qstate: {qstate}")

    dx, dy, theta1, theta2, v_own, v_int = qstate

    # dx is intruder - ownship

    # convert to network input
    rho = np.sqrt(dx*dx + dy*dy)

    if rho > 60760:
        cmd = 0
    else:
        theta = atan2(dy, dx)

        # compute thetas from velocities?
        #theta1 = atan2(vyo, vxo)
        #theta2 = atan2(vyi, vxi)
        psi = theta2 - theta1
        #print(f"dvy = {dvy}, dvx = {dvx}, psi = {psi}")

        theta -= theta1 # angle to intruder relative to ownship heading direction

        # get angles into range
        while theta < -np.pi:
            theta += 2 * np.pi

        while theta > np.pi:
            theta -= 2 * np.pi

        while psi < -np.pi:
            psi += 2 * np.pi

        while psi > np.pi:
            psi -= 2 * np.pi

        if stdout:
            print(f"qinputs: {rho, theta, psi, v_own, v_int}")

        i = np.array([rho, theta, psi, v_own, v_int])
        net = get_network(alpha_prev)
        out = run_network(net, i)
        cmd = int(np.argmin(out))

    return cmd

# TODO: check effect on runtime if LRU cache is used here
# @lru_cache(maxsize=None)
def run_network(session, x, stdout=False):
    'run the network and return the output'

    range_for_scaling, means_for_scaling = get_scaling()

    tol = 1e-4
    min_inputs = [0, -pi, -pi, 100, 0]
    max_inputs = [60760, pi, pi, 1200, 1200]

    # normalize input
    for i in range(5):
        assert min_inputs[i] - tol <= x[i] <= max_inputs[i] + tol
        
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})
        
    return outputs[0][0]

@lru_cache(maxsize=None)
def get_scaling():
    """get scaling params"""

    means_for_scaling = (19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975)
    range_for_scaling = (60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0)

    return range_for_scaling, means_for_scaling

@lru_cache(maxsize=None)
def get_network(last_cmd):
    '''load the one neural network as an ort session'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    path = os.path.join("resources", onnx_filename)
    session = ort.InferenceSession(path)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session
