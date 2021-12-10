"""
Neural network interface for quantized backreach

Stanley Bak, 12-2021
"""

import numpy as np

import onnxruntime as ort

def run_network(session, x, stdout=False):
    'run the network and return the output'

    range_for_scaling, means_for_scaling = get_scaling

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})
        
    return outputs[0][0]

def get_scaling():
    """get scaling params"""

    means_for_scaling = (19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975)
    range_for_scaling = (60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0)

    range_for_scaling, means_for_scaling

def load_network(last_cmd):
    '''load the one neural network as an ort session'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets
