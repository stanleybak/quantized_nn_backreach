"""
Neural network interface for quantized backreach

Stanley Bak, 12-2021
"""

from functools import lru_cache

import os
from math import pi, atan2

import numpy as np

import onnxruntime as ort

from timerutil import timed

from settings import Settings

# TODO: check effect on runtime if LRU cache is used here
@lru_cache(maxsize=int(1e6))
@timed
def get_cmd(alpha_prev, tau_index, qdx, qdy, qtheta1, qv_own, qv_int, stdout=False) -> int:
    """get the command at the given quantized state

    returns cmd
    """

    assert isinstance(alpha_prev, int) and 0 <= alpha_prev <= 4, f"alpha_prev was {alpha_prev}"
    assert isinstance(qdx, int)
    assert 0 <= tau_index <= 8, f"tau_index out of bounds: {tau_index}"

    pos_quantum = Settings.pos_q
    vel_quantum = Settings.vel_q
    theta1_quantum = Settings.theta1_q

    # convert quantized state to floats
    dx = pos_quantum / 2 + pos_quantum * qdx
    dy = pos_quantum / 2 + pos_quantum * qdy
    theta1 = theta1_quantum / 2 + theta1_quantum * qtheta1
    theta2 = 0

    if vel_quantum != 0:
        v_own = vel_quantum / 2 + qv_own * vel_quantum
        v_int = vel_quantum / 2 + qv_int * vel_quantum
    else:
        v_own = qv_own
        v_int = qv_int

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

        qinput = (rho, theta, psi, v_own, v_int)
            
        if stdout:
            print(f"qinput: {qinput}")

        i = np.array(qinput)

        out = run_network(alpha_prev, tau_index, i)
        cmd = int(np.argmin(out))

    return cmd

def get_cmd_continuous(alpha_prev, tau_index, dx, dy, theta1, v_own, v_int) -> int:
    """get the command at the given continuous state

    returns cmd
    """

    assert isinstance(alpha_prev, int) and 0 <= alpha_prev <= 4, f"alpha_prev was {alpha_prev}"

    # convert quantized state to floats
    theta2 = 0

    # dx is intruder - ownship

    # convert to network input
    rho = np.sqrt(dx*dx + dy*dy)

    if rho > 60760:
        cmd = 0
    else:
        theta = atan2(dy, dx)

        psi = theta2 - theta1

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

        net_input = (rho, theta, psi, v_own, v_int)
            
        i = np.array(net_input)

        out = run_network(alpha_prev, tau_index, i)
        cmd = int(np.argmin(out))

    return cmd

@timed
def run_network(alpha_prev, tau_index, x, stdout=False):
    'run the network and return the output'

    # cached
    session = get_network(alpha_prev, tau_index)

    range_for_scaling, means_for_scaling = get_scaling()

    tol = 1e-2
    min_inputs = [0, -pi, -pi, 100, 0]
    max_inputs = [60760, pi, pi, 1200, 1200]

    # normalize input
    for i in range(5):
        assert min_inputs[i] - tol <= x[i] <= max_inputs[i] + tol, f"network input input {i} out of bounds. got " + \
            f"{x[i]}, valid range: {min_inputs[i], max_inputs[i]}"
        
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
def get_network(alpha_prev, tau_index):
    '''load the one neural network as an ort session'''

    onnx_filename = f"ACASXU_run2a_{alpha_prev + 1}_{tau_index + 1}_batch_2000.onnx"

    path = os.path.join("resources", onnx_filename)
    session = ort.InferenceSession(path)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session
