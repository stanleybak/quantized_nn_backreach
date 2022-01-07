"""
code related to paralell backreachability

Stanley Bak, Jan 2021
"""

from typing import List, Tuple
import time
import multiprocessing
import pickle
from math import pi

from settings import pos_quantum, vel_quantum, theta1_quantum
from util import to_time_str, get_num_cores, is_init_qx_qy
from timerutil import Timers

global_start_time = 0.0
global_params_list: List[Tuple[int, int, int, int, int, int]] = [] # assigned after we get params made
global_process_id = -1 # assigned in init

shared_next_index = multiprocessing.Value('i', 0) # the next index to be done
shared_cur_index = multiprocessing.Array('i', get_num_cores()) # the current index for each core
shared_cur_index_start_time = multiprocessing.Array('d', get_num_cores()) # the start time for the current job

shared_num_counterexamples = multiprocessing.Value('i', 0)

def increment_index() -> Tuple[int, Tuple[int, int, int, int, int, int]]:
    """get next work index and params (and print update)"""

    next_index = -1
    now = time.perf_counter()

    with shared_next_index.get_lock():
        next_index = shared_next_index.value
        shared_next_index.value += 1

        shared_cur_index[global_process_id] = next_index
        shared_cur_index_start_time[global_process_id] = now

    if next_index % 10000 == 0:
        # print longest-running job
        longest: Tuple[float, int] = (now - shared_cur_index_start_time[0], 0)

        for i in range(get_num_cores()):
            cur: Tuple[float, int] = (now - shared_cur_index_start_time[i], i)

            if cur > longest:
                longest = cur

        runtime, index = longest

        print(f"\nLongest Running Job: index={index} ({round(runtime, 2)}s)")

        # print progress
        num_cases = len(global_params_list)
        percent = 100 * next_index / num_cases
        elapsed = time.perf_counter() - global_start_time
        eta = elapsed * num_cases / (next_index + 1) - elapsed

        print(f"{round(percent, 2)}% Elapsed: {to_time_str(elapsed)}, ETA: {to_time_str(eta)} " + \
              f"{next_index}/{num_cases}: ", end='', flush=True)

    if next_index % 200 == 0:
        print(".", end='', flush=True)

    #params = global_params_list[next_index]
    #print(next_index, params)

    return next_index, params

def make_params():
    """make params for parallel run"""

    vel_ownship = (100, 1200)
    #vel_intruder = (0, 1200) # full range
    vel_intruder = (0, 400)

    assert -500 % pos_quantum < 1e-6
    assert 500 % pos_quantum < 1e-6

    x_own_min = round(-500 / pos_quantum)
    x_own_max = round(500 / pos_quantum)

    y_own_min = round(-500 / pos_quantum)
    y_own_max = round(500 / pos_quantum)

    max_qtheta1 = round(2*pi / theta1_quantum)

    qvimin = round(vel_intruder[0]/vel_quantum)
    qvimax = round(vel_intruder[1]/vel_quantum)

    qvomin = round(vel_ownship[0]/vel_quantum)
    qvomax = round(vel_ownship[1]/vel_quantum)

    for i in range(2):
        assert vel_ownship[i] % vel_quantum < 1e-6
        assert vel_intruder[i] % vel_quantum < 1e-6

    params_list = []

    # try to do cases that are more likely to be false first
    for alpha_prev in reversed(range(5)):
        for q_vint in reversed(range(qvimin, qvimax)):
            for q_vown in range(qvomin, qvomax):
                for y_own in range(y_own_min, y_own_max):
                    for x_own in range(x_own_min, x_own_max):

                        if not is_init_qx_qy(x_own, y_own):
                            continue

                        for qtheta1 in range(0, max_qtheta1):
                            params = (alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint)
                            params_list.append(params)

    return params_list

def init_process(q):
    """init the process"""

    global global_process_id
    global_process_id = q.get()

    Timers.enabled = False

    print(f"in init_process, len(global_params_list) = {len(global_params_list)}")

def print_result(label, res):
    """print info on the passed-in backreach result"""

    diff = res['runtime']
    index = res['index]']
    alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint = global_params_list[index]

    num_popped = res['num_popped']
    unique_paths = res['unique_paths']
    unsafe = res['counterexample'] is not None

    print(f"{label} ({round(diff, 2)}) for:\nalpha_prev={alpha_prev}\nx_own={x_own}\n" + \
          f"y_own={y_own}\nqtheta1={qtheta1}\nq_vown={q_vown}\nq_vint={q_vint}")
    print(f'num_popped: {num_popped}, unique_paths: {unique_paths}, has_counterexample: {unsafe}')

def get_counterexamples(backreach_single):
    """get all counterexamples at the current quantization"""

    global global_start_time
    global global_params_list

    # shared variable for coordination
    start = time.perf_counter()
    global_params_list = make_params()
    diff = time.perf_counter() - start
    
    num_cases = len(global_params_list)
    print(f"Made params for {num_cases} cases in {round(diff, 2)} secs")
    
    global_start_time = time.perf_counter()
    q = multiprocessing.Queue()

    for i in range(get_num_cores()):
        q.put(i)

    counterexamples = []
    total_runtime = 0.0

    with multiprocessing.Pool(get_num_cores(), initializer=init_process, initargs=(q, )) as pool:
        res_list = pool.map(backreach_single, range(num_cases))
        max_runtime = res_list[1]

        for res in res_list:
            if res['counterexample'] is not None:
                counterexamples.append(res)

            t = res['runtime']
            total_runtime += t

            if t > max_runtime['runtime']:
                max_runtime = res

    diff = time.perf_counter() - global_start_time
    print(f"\nfinished enumeration ({num_cases} cases) in {to_time_str(diff)}\n" + \
          f"({len(counterexamples)}) counterexamples")
    print(f"Avg runtime per case: {to_time_str(total_runtime / num_cases)}")

    return counterexamples, max_runtime

def save_counterexamples(counterexamples, filename):
    """pickle and save all counterexamples"""

    raw = pickle.dumps(counterexamples)
    mb = len(raw) / 1024 / 1024

    with open(filename, "wb") as f:
        f.write(raw)

    print(f"Saved {len(counterexamples)} counterexamples ({round(mb, 3)} MB) to {filename}")

def run_all_parallel(backreach_single):
    """loop over all cases"""

    counterexamples, max_runtime = get_counterexamples(backreach_single)

    print()
    print_result('longest runtime', max_runtime)

    for i, counterexample_res in enumerate(counterexamples):
        counterexample = counterexample_res['counterexample']
        
        print("\nCounterexample {i}:")
        counterexample.print_replay_init()
        counterexample.print_replay_witness(plot=False)

        print_result(f"Counterexample {i}", counterexample_res)
        
    if counterexamples:
        print("\nIncomplete analysis; had counterexamples.")

        save_counterexamples(counterexamples, 'counterexamples.pkl')
    else:
        print("\nDone! No counterexamples.")
