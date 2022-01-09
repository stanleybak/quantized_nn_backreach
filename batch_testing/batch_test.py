"""
batch test code for onnx

measure performance of batch execution of networks
"""

import time
import numpy as np

import onnxruntime as ort

def get_network():
    '''load the one neural network as an ort session'''

    onnx_filename = f"../resourcesACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    path = os.path.join("resources", onnx_filename)
    session = ort.InferenceSession(path)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session

def main():
    """main entry point"""

    path = "acasxu1.onnx"
    session = ort.InferenceSession(path)

    i = np.array([0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    iterations = 100000
    start = time.perf_counter()
    outputs = []
    
    for _ in range(iterations):
        outputs.append(session.run(None, {'input': i}))

    diff = time.perf_counter() - start
    print(f"loop runtime: {diff}")

    ################
    path = "acasxu1_batched.onnx"
    session = ort.InferenceSession(path)

    batch_size = 100
    i = np.array([0, 0.1, 0.2, 0.3, 0.4] * batch_size, dtype=np.float32)
    i.shape = (batch_size, 1, 1, 5)

    start = time.perf_counter()
    outputs = []
    
    for _ in range(round(iterations/batch_size)):
        batch_outputs = session.run(None, {'input': i})

        print(batch_outputs[0].shape)
        exit(1)

    diff = time.perf_counter() - start
    print(f"batch2 runtime: {diff}")

if __name__ == "__main__":
    main()
