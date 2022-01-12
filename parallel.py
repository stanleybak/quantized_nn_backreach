"""
code related to paralell backreachability

Stanley Bak, Jan 2021
"""

from typing import List, Tuple
import time
import multiprocessing
import pickle
from math import pi, atan2, sqrt

import matplotlib.pyplot as plt

from settings import Settings
from util import to_time_str, get_num_cores, is_init_qx_qy, get_tau_index
from timerutil import Timers
from star import Star
from networks import get_cmd_continuous
from dubins import get_time_elapse_mat

global_start_time = 0.0
global_params_list: List[Tuple[int, int, int, int, int, int]] = [] # assigned after we get params made
global_process_id = -1 # assigned in init

shared_next_index = multiprocessing.Value('i', 0) # the next index to be done
shared_next_print_time = multiprocessing.Value('f', 0)
shared_cur_index = multiprocessing.Array('i', get_num_cores()) # the current index for each core
shared_cur_index_start_time = multiprocessing.Array('d', get_num_cores()) # the start time for the current job
shared_num_counterexamples = multiprocessing.Value('i', 0)
shared_num_timeouts = multiprocessing.Value('i', 0)
shared_counterexamples_list = multiprocessing.Manager().list()

def increment_index() -> Tuple[int, Tuple[int, int, int, int, int, int]]:
    """get next work index and params (and print update)"""

    next_index = -1
    now = time.perf_counter()

    with shared_next_index.get_lock():
        next_index = shared_next_index.value
        shared_next_index.value += 1

        shared_cur_index[global_process_id] = next_index
        shared_cur_index_start_time[global_process_id] = now

        if next_index == 0: # reset all times
            for i in range(get_num_cores()):
                shared_cur_index_start_time[i] = now
                shared_cur_index[i] = 0
                shared_next_print_time.value = now

        if now >= shared_next_print_time.value:
            print_increment_secs = 5
            shared_next_print_time.value = now + print_increment_secs

            if next_index > 0:
                # print longest-running job
                longest: Tuple[float, int] = (now - shared_cur_index_start_time[0], 0)

                for i in range(get_num_cores()):
                    index = shared_cur_index[i]
                    runtime = now - shared_cur_index_start_time[i]
                    cur: Tuple[float, int] = (runtime, index)

                    if cur > longest:
                        longest = cur

                runtime, index = longest

                if len(shared_counterexamples_list) > 0:
                    print(f"\nTimeouts: {shared_num_timeouts.value}, Counterexamples " + \
                          f"({len(shared_counterexamples_list)}): {shared_counterexamples_list}",
                          end='')

                print(f"\nLongest Running Job: index={index} ({to_time_str(runtime)})")

            # print progress
            num_cases = len(global_params_list)
            percent = 100 * next_index / num_cases
            elapsed = time.perf_counter() - global_start_time
            eta = elapsed * num_cases / (next_index + 1) - elapsed

            print(f"{round(percent, 2)}% Elapsed: {to_time_str(elapsed)}, ETA: {to_time_str(eta)} " + \
                  f"{next_index}/{num_cases}: ", end='', flush=True)

    if next_index % 100 == 0:
        print(".", end='', flush=True)

    params = global_params_list[next_index]

    if Settings.max_counterexamples is not None and shared_num_counterexamples.value > Settings.max_counterexamples:
        next_index = -1

    return next_index, params

def worker_had_counterexample(res):
    """called when worker has a counterexample to update shared state"""

    shared_counterexamples_list.append(res['index'])

def make_params(max_index=None):
    """make params for parallel run"""

    # 1000-1200 is safe with 250, 50, 1.5, 950-1000 is not!
    # finished enumeration (1843200 cases) in 13.68 min, counterexamples: 0
    #num timeouts: 0
    #Avg runtime per case: 0.055 sec
    #longest runtime (15.23) (index: 949)

    
    # 900-1000 safe with 250, 25, 1.5, but not 850-900!
    #finished enumeration (3686400 cases) in 23.93 min, counterexamples: 0
    #num timeouts: 0
    #Avg runtime per case: 0.048 sec
    #longest runtime (58.26) (index: 693)
    
    # 881.5 has unsafe case!!!!!!
    
    vel_ownship = Settings.range_vel_ownship
    vel_intruder = Settings.range_vel_intruder #(0, 1200) # full range
    #vel_intruder = (0, 300)

    print(f"Making params with vel_ownship={vel_ownship}, vel_intruder={vel_intruder}, " + \
          f"and max_index={max_index}...")

    pos_quantum = Settings.pos_q
    vel_quantum = Settings.vel_q
    theta1_quantum = Settings.theta1_q

    assert -500 % pos_quantum < 1e-6
    assert 500 % pos_quantum < 1e-6

    x_own_min = round(-500 / pos_quantum)
    x_own_max = round(500 / pos_quantum)

    y_own_min = round(-500 / pos_quantum)
    y_own_max = round(500 / pos_quantum)

    max_qtheta1 = round(2*pi / theta1_quantum)

    if vel_quantum != 0:
        qvimin = round(vel_intruder[0]/vel_quantum)
        qvimax = round(vel_intruder[1]/vel_quantum)

        qvomin = round(vel_ownship[0]/vel_quantum)
        qvomax = round(vel_ownship[1]/vel_quantum)

        for i in range(2):
            assert vel_ownship[i] % vel_quantum < 1e-6
            assert vel_intruder[i] % vel_quantum < 1e-6
    else:
        qvimin = vel_intruder[0]
        qvimax = vel_intruder[0] + 1

        qvomin = vel_ownship[0]
        qvomax = vel_ownship[0] + 1

    params_list = []

    # try to do cases that are more likely to be false first
    for alpha_prev in reversed(range(5)):
        if max_index is not None and len(params_list) > max_index:
            break
        
        for q_vint in reversed(range(qvimin, qvimax)):
            if max_index is not None and len(params_list) > max_index:
                break
        
            for q_vown in range(qvomin, qvomax):
                if max_index is not None and len(params_list) > max_index:
                    break
        
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

def print_result(label, res):
    """print info on the passed-in backreach result"""

    diff = res['runtime']
    index = res['index']
    alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint = global_params_list[index]

    num_popped = res['num_popped']
    unique_paths = res['unique_paths']
    unsafe = res['counterexample'] is not None

    print(f"{label} ({round(diff, 2)}) (index: {index}) for:\nalpha_prev={alpha_prev}\nx_own={x_own}\n" + \
          f"y_own={y_own}\nqtheta1={qtheta1}\nq_vown={q_vown}\nq_vint={q_vint}")
    print(f'num_popped: {num_popped}, unique_paths: {unique_paths}, has_counterexample: {unsafe}')

    if unsafe:
        rad = res['counterexample'].star.get_witness(get_radius=True)[-1]
        print(f"chebeshev radius of witness: {rad}")

def get_counterexamples(backreach_single, indices=None, params=None):
    """get all counterexamples at the current quantization"""

    global global_start_time
    global global_params_list

    # reset values
    shared_next_index.value = 0 
    shared_num_counterexamples.value = 0
    shared_num_timeouts.value = 0
    shared_counterexamples_list[:] = [] # clear list

    if params is not None:
        num_cases = len(params)
        global_params_list = params
        print(f"Using passed-in params (num: {num_cases})")

    else:
        max_index = None

        if indices is not None:
            max_index = max(indices)
        
        start = time.perf_counter()
        global_params_list = make_params(max_index)
        diff = time.perf_counter() - start

        if indices is not None:
            print(f"WARNING: using fixed indices up to max_index={max_index}")
            global_params_list = [global_params_list[i] for i in indices]

        num_cases = len(global_params_list)
        print(f"Made params for {num_cases} cases in {round(diff, 2)} secs")
        
    global_start_time = time.perf_counter()
    q = multiprocessing.Queue()

    for i in range(get_num_cores()):
        q.put(i)

    counterexamples = []
    total_runtime = 0.0

    res_was_none = False

    with multiprocessing.Pool(get_num_cores(), initializer=init_process, initargs=(q, )) as pool:
        res_list = pool.map(backreach_single, range(num_cases), chunksize=1)
        max_runtime = res_list[0]
        timeouts = 0

        for res in res_list:
            if res is None:
                res_was_none = True
                continue
            
            if res['counterexample'] is not None:
                counterexamples.append(res)

            if res['timeout']:
                counterexamples.append(res)
                timeouts += 1

            t = res['runtime']
            total_runtime += t

            if t > max_runtime['runtime']:
                max_runtime = res

    diff = time.perf_counter() - global_start_time
    print(f"\nfinished enumeration ({num_cases} cases) in {to_time_str(diff)}, " + \
          f"counterexamples: {len(counterexamples)}")

    print(f"num timeouts: {timeouts}")

    if shared_counterexamples_list:
        print(f"counterexamples ({len(shared_counterexamples_list)}): {shared_counterexamples_list}")
    
    print(f"Avg runtime per case: {to_time_str(total_runtime / num_cases)}")

    if res_was_none:
        print("Warning: Incomplete analysis: res was None for some cases (number of counterexamples " + \
              f"exceeded Settings.max_counterexamples={Settings.max_counterexamples})")

    return counterexamples, max_runtime

def save_counterexamples(counterexamples, filename):
    """pickle and save all counterexamples"""

    print("saving counterexamples...")

    raw = pickle.dumps(counterexamples)
    mb = len(raw) / 1024 / 1024

    with open(filename, "wb") as f:
        f.write(raw)

    print(f"Saved {len(counterexamples)} counterexamples ({round(mb, 3)} MB) to {filename}")

def is_real_counterexample(res):
    """is the passed in backreach a real (non-quantized counter-example)"""

    assert res['counterexample'] is not None

    s = res['counterexample']

    _, range_pt, radius = s.star.get_witness(get_radius=True)
    tau = s.tau

    if radius < 1e-6:
        #print(f"chebeshev radius was too small ({radius}), skipping replay")
        return False
    
    pt = range_pt.copy()

    #q_theta1 = s.qtheta1
    #s_copy = deepcopy(s)

    #pos_quantum = Settings.pos_q
    #mismatch_quantized = False
    #mismatch_continuous = False
    unsafe_continuous = False
    c_cmd_out = s.alpha_prev_list[-1]
    assert c_cmd_out == 0

    if Settings.tau_dot != 0:
        assert tau + 1 == len(s.alpha_prev_list)

    for _ in s.alpha_prev_list:
        #expected_cmd = s.alpha_prev_list[-(i+2)]

        dx = pt[Star.X_INT] - pt[Star.X_OWN]
        dy = 0 - pt[Star.Y_OWN]
        
        #qdx = floor(dx / pos_quantum)
        #qdy = floor(dy / pos_quantum)

        rho = sqrt(dx*dx + dy*dy)

        if rho < 500 and tau == 0:
            print(f"Non-quantized replay is unsafe (rho={rho})! Real counterexample.")
            print()
            s.print_replay_init()
            print()
        
            unsafe_continuous = True
            break

        #qstate = (qdx, qdy, q_theta1, s.qv_own, s.qv_int)
        #q_cmd_out = get_cmd(net, *qstate)

        c_theta1 = atan2(pt[Star.VY_OWN], pt[Star.VX_OWN])
        #quantized = q_theta1 * Settings.theta1_q + Settings.theta1_q / 2
        
        #vown = s.qv_own * Settings.vel_q + Settings.vel_q / 2
        #vint = s.qv_int * Settings.vel_q + Settings.vel_q / 2
        vown = sqrt(pt[Star.VX_OWN]**2 + pt[Star.VY_OWN]**2)
        vint = sqrt(pt[Star.VX_INT]**2 + 0**2)
        cstate = (dx, dy, c_theta1, vown, vint)

        tau_index = get_tau_index(tau)
        c_cmd_out = get_cmd_continuous(c_cmd_out, tau_index, *cstate)

        #print(f"{i}. net: {net}, cstate: {cstate}, c_cmd: {c_cmd_out}")
        
        #if q_cmd_out != expected_cmd and not mismatch_quantized:
        #    #print(f"Quantized mismatch at step {i+1}. got cmd {q_cmd_out}, expected cmd {expected_cmd}")
         #   mismatch_quantized = True

        #if c_cmd_out != expected_cmd and not mismatch_continuous:
            #print(f"Non-quantized mismatch at step {i+1}/{len(s.alpha_prev_list) - 1}. " + \
            #      f"got cmd {c_cmd_out}, expected cmd {expected_cmd}")
        #    mismatch_continuous = True

        #if mismatch_quantized and mismatch_continuous:
        #    break

        #s_copy.backstep(forward=True, forward_alpha_prev=expected_cmd)
        #s_copy.backstep(forward=True, forward_alpha_prev=c_cmd_out)

        mat = get_time_elapse_mat(c_cmd_out, 1.0)
        pt = mat @ pt
        tau += Settings.tau_dot

        #delta_q_theta = Settings.cmd_quantum_list[expected_cmd]# * theta1_quantum
        #q_theta1 += delta_q_theta

    #if mismatch_quantized:
    #    print("Warning: Quantized replay DOES NOT match")

    return unsafe_continuous

def refine_counterexamples(backreach_single, counterexamples, level=0):
    """refine counterexamples

    returns True if refinement is safe
    """

    print(f"\n####### level {level}: Refining {len(counterexamples)} counterexamples ######")

    # need to do this check before refining Settings
    print("Replaying counterexamples...")
    for i, counterexample in enumerate(counterexamples):
        #print(f"Replaying counterexample {i+1}/{len(counterexamples)}")

        if counterexample['timeout']:
            #print("was timeout")
            continue
        
        assert counterexample['counterexample'] is not None
        
        if is_real_counterexample(counterexample):
            print(f"Found reach counterexample at state: {counterexample['params']}")
            return False

    qstates = [] # qstates after refinement

    levels = ['pos', 'vel', 'pos', 'vel', 'pos', 'vel', 'theta1', 'pos', 'vel', 'theta1',
              'pos', 'vel', 'theta1', 'pos', 'vel', 'pos', 'vel', 'pos', 'vel']

    if level >= len(levels):
        print(f"Refinement reached max level: {len(levels)}")
        return False
    
    if levels[level] == 'pos' or (levels[level] == 'vel' and Settings.vel_q == 0):
        print(f"Level {level}: refining q_pos from {Settings.pos_q} to {Settings.pos_q / 2}")
        Settings.pos_q /= 2
    elif levels[level] == 'vel':
        print(f"Level {level}: refining q_vel from {Settings.vel_q} to {Settings.vel_q / 2}")
        Settings.vel_q /= 2
    elif levels[level] == 'theta1':
        print(f"Level {level}: refining q_theta1 from {Settings.theta1_q_deg} to {Settings.theta1_q_deg / 2}")
        Settings.theta1_q_deg /= 2
        Settings.theta1_q /= 2
        Settings.init_cmd_quantum_list() # since theta1 was changed

    Settings.single_case_timeout *= 2

    print(f"Level {level} with Settings: pos={Settings.pos_q}, vel={Settings.vel_q}, theta1={Settings.theta1_q_deg}, " + \
          f"timeout={Settings.single_case_timeout}")

    for counterexample in counterexamples:
        alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint = counterexample['params']
        s = counterexample['counterexample']
        
        #if s is not None:
        #    _, _, radius = s.star.get_witness(get_radius=True)
        #    print(f"rad: {radius}")

        #    if radius < 1e-6: # radius was too small to be a real counterexample (replay fails due to numerics)
        #        continue

        cqstates = [] # candidate qstates for the next iteration

        # add refined states to qstates
        if levels[level] == 'pos':
            cqstates.append((alpha_prev, 2*x_own, 2*y_own, qtheta1, q_vown, q_vint))
            cqstates.append((alpha_prev, 2*x_own + 1, 2*y_own, qtheta1, q_vown, q_vint))
            cqstates.append((alpha_prev, 2*x_own, 2*y_own + 1, qtheta1, q_vown, q_vint))
            cqstates.append((alpha_prev, 2*x_own + 1, 2*y_own + 1, qtheta1, q_vown, q_vint))
        elif levels[level] == 'vel':
            cqstates.append((alpha_prev, x_own, y_own, qtheta1, 2*q_vown, 2*q_vint))
            cqstates.append((alpha_prev, x_own, y_own, qtheta1, 2*q_vown+1, 2*q_vint))
            cqstates.append((alpha_prev, x_own, y_own, qtheta1, 2*q_vown, 2*q_vint+1))
            cqstates.append((alpha_prev, x_own, y_own, qtheta1, 2*q_vown+1, 2*q_vint+1))
        else:
            assert levels[level] == 'theta1'
            cqstates.append((alpha_prev, x_own, y_own, 2*qtheta1, q_vown, q_vint))
            cqstates.append((alpha_prev, x_own, y_own, 2*qtheta1 + 1, q_vown, q_vint))

        for qstate in cqstates:
            x_own = qstate[1]
            y_own = qstate[2]

            # make sure it's still an initial state after refinement
            if is_init_qx_qy(x_own, y_own):
                qstates.append(qstate)

    rv = True
        
    if not qstates:
        print("no more qstates after trimming initial states, safe!")
    else:
        if len(qstates) > Settings.max_counterexamples:
            qstates = qstates[:Settings.max_counterexamples]
        
        new_counterexamples, _ = get_counterexamples(backreach_single, params=qstates)

        if new_counterexamples:
            rv = refine_counterexamples(backreach_single, new_counterexamples, level=level + 1)
        else:
            print("no new counterexamples found, safe!")

    return rv

def run_all_parallel(backreach_single, tau_dot, indices=None):
    """loop over all cases, returns true if proven safe witout refining"""

    assert tau_dot in [-1, 0]
    Settings.tau_dot = tau_dot
    
    counterexamples, max_runtime = get_counterexamples(backreach_single, indices=indices)

    print()
    print_result('longest runtime', max_runtime)

    #for i, counterexample_res in enumerate(counterexamples):
    #    counterexample = counterexample_res['counterexample']
        
    #    print(f"\nCounterexample {i}:")
    #    counterexample.print_replay_init()
    #    counterexample.print_replay_witness(plot=False)

    #    print_result(f"Counterexample {i}", counterexample_res)

    if indices is None:
        if counterexamples:
            print("\nIncomplete analysis; had counterexamples.")

            #save_counterexamples(counterexamples, 'counterexamples.pkl')
        else:
            print("\nDone! No counterexamples.")
    else:
        print(f"Finished passed-in {len(indices)} indices.")

    if not counterexamples:
        print("original system had no counterexamples")
        rv = True
    else:
        rv = False
        counterexamples_safe = refine_counterexamples(backreach_single, counterexamples)

        if counterexamples_safe:
            print("counterexamples proven safe after refining")
        else:
            print("counterexamples were not proven safe")

    return rv

def refine_indices(backreach_single, counterexample_index_list):
    """a debuggin function, refine a specific set of counterexample indices"""

    assert isinstance(counterexample_index_list, list)
    assert counterexample_index_list, "empty list of counterexamples?"
    assert isinstance(counterexample_index_list[0], int)
    
    max_index = max(counterexample_index_list)

    start = time.perf_counter()
    params = make_params(max_index)
    num_cases = len(params)
    diff = time.perf_counter() - start

    print(f"Made params for {num_cases} cases in {round(diff, 2)} secs")

    params = [params[index] for index in counterexample_index_list]

    new_counterexamples, _ = get_counterexamples(backreach_single, params=params)
    rv = True
    
    if new_counterexamples:
        rv = refine_counterexamples(backreach_single, new_counterexamples)

    print(f"Finished refine_indices. Result was safe={rv}")

def run_single_case(backreach_single, index, plot=False):
    """test a single (difficult) case"""

    print(f"running single with index={index}...")

    start = time.perf_counter()
    params = make_params(index+1)
    num_cases = len(params)
    diff = time.perf_counter() - start

    print(f"Made params for {num_cases} cases in {round(diff, 2)} secs")

    tup = params[index]

    Timers.tic('top')
    res = backreach_single(tup, parallel=False, plot=plot)
    Timers.toc('top')
    Timers.print_stats()

    if res is not None:
        print(f"popped: {res['num_popped']}")
        print(f"unique_paths: {res['unique_paths']}")
        print(f"counterexample: {res['counterexample'] is not None}")
        print(f"timeout: {res['timeout']}")

    if plot:
        plt.show()
