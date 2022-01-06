'''
Backreach using quantized inputs
'''

from typing import List, Tuple, Dict, TypedDict, Optional, Any

import time
from copy import deepcopy
from math import pi, floor, ceil

import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from star import Star
from plotting import Plotter
from dubins import init_to_constraints, get_time_elapse_mat
from util import make_qstar, to_time_str
from networks import get_cmd

from timerutil import timed, Timers
from settings import pos_quantum, vel_quantum, theta1_quantum

shared_num_counterexamples = multiprocessing.Value('i', 0) # for syncing multiple processes to exit
shared_num_completed = multiprocessing.Value('i', 0) # for syncing multiple processes to print updated progress
global_total_num_cases = -1 # gets set once
global_start_time = 0.0

def increment_progress():
    """increment global progress (and possibly print update)"""

    completed = -1

    with shared_num_completed.get_lock():
        shared_num_completed.value += 1
        completed = shared_num_completed.value
    
    if completed % 50 == 1:
        # print progress
        
        percent = 100 * completed / global_total_num_cases
        elapsed = time.perf_counter() - global_start_time
        eta = elapsed * global_total_num_cases / (completed + 1) - elapsed

        print(f"\n{round(percent, 2)}% Elapsed: {to_time_str(elapsed)}, ETA: {to_time_str(eta)} " + \
              f"{completed}/{global_total_num_cases}: ", end='', flush=True)
    else:
        print(".", end='', flush=True)

class State():
    """state of backreach container

    state is:
    alpha_prev (int) - previous command 
    """

    debug = False

    nn_update_rate = 1.0
    next_state_id = 0

    cmd_quantum_list: List[int] = [] # [0, 1, -1, 2, -2]

    @classmethod
    def init_class(cls):
        '''init class variables'''
        
        q = 2*pi / (360 / 1.5)
        assert theta1_quantum * round(q/theta1_quantum) - q < 1e-6
        assert not State.cmd_quantum_list

        # initialize
        q = 2*pi / (360 / 1.5)
        cls.cmd_quantum_list = [0, round(q/theta1_quantum), -1 * round(q/theta1_quantum),
                                         2 * round(q/theta1_quantum), -2 * round(q/theta1_quantum)]

    def __init__(self, alpha_prev: int, qtheta1: int, qv_own: int, qv_int: int, \
                 star: Star):

        assert isinstance(qtheta1, int)
        self.qtheta1 = qtheta1
        self.qv_own = qv_own
        self.qv_int = qv_int      
        
        self.alpha_prev_list = [alpha_prev]

        if State.debug:
            self.qstate_star_list = []
        else:
            self.qstate_star_list = None
        
        self.star = star
        
        self.state_id = -1
        self.assign_state_id()

    def __str__(self):
        return f"State(id={self.state_id} with alpha_prev_list = {self.alpha_prev_list})"

    @timed
    def copy(self, new_star=None):
        """return a deep copy of self"""

        if new_star is not None:
            self_star = self.star
            self.star = None

        rv = deepcopy(self)
        rv.assign_state_id()

        if new_star is not None:
            rv.star = new_star
            
            # restore self.star
            self.star = self_star

        return rv

    def print_replay_init(self):
        """print initialization states for replay"""

        print(f"alpha_prev_list = {self.alpha_prev_list}")
        print(f"qtheta1 = {self.qtheta1}")
        print(f"qv_own = {self.qv_own}")
        print(f"qv_int = {self.qv_int}")

    def print_replay_witness(self, plot=False):
        """print a step-by-step replay for the witness point"""

        s = self
        
        domain_pt, range_pt = s.star.get_witness(print_radius=True)
        print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

        p = Plotter()

        print()

        if State.debug:
            step = 0
            for i in range(len(s.qstate_star_list) - 1, -1, -1):
                step += 1
                qstate, star = s.qstate_star_list[i]

                print(f"{step}. network {s.alpha_prev_list[i+1]} with qstate: {qstate} " + \
                      f"gave cmd {s.alpha_prev_list[i]}")

                #pt = star.get_witness(print_radius=True)[1]
                # quantize witness
                #dx = floor((pt[Star.X_INT] - pt[Star.X_OWN]) / pos_quantum)
                #dy = floor((0 - pt[Star.Y_OWN]) / pos_quantum)
                #print(f"quantized witness: {dx, dy}")

                p.plot_star(star, color='magenta')

            print()

        pt = range_pt.copy()
        
        q_theta1 = s.qtheta1
        s_copy = deepcopy(s)

        p.plot_star(s.star, color='r')
        mismatch = False

        for i in range(len(s.alpha_prev_list) - 1):
            net = s.alpha_prev_list[-(i+1)]
            expected_cmd = s.alpha_prev_list[-(i+2)]

            dx = floor((pt[Star.X_INT] - pt[Star.X_OWN]) / pos_quantum)
            dy = floor((0 - pt[Star.Y_OWN]) / pos_quantum)

            qstate = (dx, dy, q_theta1, s.qv_own, s.qv_int)
            
            cmd_out = get_cmd(net, *qstate)
            print(f"({i+1}). network {net} -> {cmd_out}")
            print(f"state: {list(pt)}")
            print(f"qstate: {qstate}")

            if cmd_out != expected_cmd:
                print(f"Mismatch at step {i+1}. got cmd {cmd_out}, expected cmd {expected_cmd}")
                mismatch = True
                break

            s_copy.backstep(forward=True, forward_alpha_prev=cmd_out)
            p.plot_star(s_copy.star)
                
            mat = get_time_elapse_mat(cmd_out, 1.0)
            pt = mat @ pt

            delta_q_theta = State.cmd_quantum_list[cmd_out]# * theta1_quantum
            q_theta1 += delta_q_theta

        if mismatch or plot:
            plt.show()
        else:
            print("witness commands all matched expectation")

    def assign_state_id(self):
        """assign and increment state_id"""

        self.state_id = State.next_state_id
        State.next_state_id += 1

    @timed
    def backstep(self, forward=False, forward_alpha_prev=-1):
        """step backwards according to alpha_prev"""

        if forward:
            cmd = forward_alpha_prev
        else:
            cmd = self.alpha_prev_list[-1]

        assert 0 <= cmd <= 4

        mat = get_time_elapse_mat(cmd, -1.0 if not forward else 1.0)

        #if forward:
        #    print(f"condition number of transform mat: {np.linalg.cond(mat)}")
        #    print(f"condition number of a_mat: {np.linalg.cond(self.star.a_mat)}")

        self.star.a_mat = mat @ self.star.a_mat
        self.star.b_vec = mat @ self.star.b_vec

        # clear, weak left, weak right, strong left, strong right
        delta_q_theta = State.cmd_quantum_list[cmd]

        if forward:
            self.qtheta1 += delta_q_theta
        else:
            self.qtheta1 -= delta_q_theta

    @timed
    def get_predecessors(self, plotter=None, stdout=False):
        """get the valid predecessors of this star
        """

        dx_qrange, dy_qrange = self.get_dx_dy_qrange(stdout=stdout)

        # compute previous state
        self.backstep()

        if plotter is not None:
            plotter.plot_star(self.star)

        rv: List[State] = []
        dx_qrange, dy_qrange = self.get_dx_dy_qrange(stdout=stdout)

        # pass 1: do all quantized states classify the same (correct way)?
        # alternatively: are they all incorrect?

        # pass 2: if partial correct / incorrect, are any of the incorrect ones feasible?
        # if so, we may need to split the set

        constants = (self.qtheta1, self.qv_own, self.qv_int)

        for prev_cmd in range(5):
            qstate_to_cmd: Dict[Tuple[int, int], int] = {}
            all_right = True
            all_wrong = True
            correct_qstates = []
            incorrect_qstates = []
            
            for qdx in range(dx_qrange[0], dx_qrange[1] + 1):
                for qdy in range(dy_qrange[0], dy_qrange[1] + 1):
                    qstate = (qdx, qdy)

                    # skip predecessors that are also initial states
                    if is_init_qstate(qstate):
                        continue

                    out_cmd = get_cmd(prev_cmd, *qstate, *constants)

                    qstate_to_cmd[qstate] = out_cmd

                    if out_cmd == self.alpha_prev_list[-1]:
                        # command is correct
                        correct_qstates.append(qstate)
                        all_wrong = False
                    else:
                        # command is wrong
                        incorrect_qstates.append(qstate)
                        all_right = False

            if all_wrong:
                continue
            
            # for the current prev_command, was either all correct or all wrong?
            if all_right:
                prev_s = self.copy()
                prev_s.alpha_prev_list.append(prev_cmd)

                if State.debug:
                    tup = (correct_qstates, deepcopy(prev_s.star))
                    prev_s.qstate_star_list.append(tup)
                
                rv.append(prev_s)
                continue

            # still have a chance at all correct: if incorrect state are infeasible
            all_incorrect_infeasible = True

            for qstate in incorrect_qstates:
                # if incorrect commands
                star = make_qstar(self.star, qstate)

                if star.is_feasible():
                    all_incorrect_infeasible = False
                    break

            if all_incorrect_infeasible:
                prev_s = self.copy()
                prev_s.alpha_prev_list.append(prev_cmd)

                if State.debug:
                    tup = (correct_qstates, deepcopy(prev_s.star))
                    prev_s.qstate_star_list.append(tup)
                
                rv.append(prev_s)
                continue

            # ok, do splitting if command is correct
            for qstate in correct_qstates:
                star = make_qstar(self.star, qstate)

                if star.is_feasible():
                    prev_s = self.copy(star)
                    prev_s.alpha_prev_list.append(prev_cmd)

                    if State.debug:
                        tup = (qstate, deepcopy(prev_s.star))
                        prev_s.qstate_star_list.append(tup)

                    rv.append(prev_s)

        return rv

    @timed
    def get_dx_dy_qrange(self, stdout=False) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """get the quantized range for (dx, dy)"""

        vec = np.zeros(Star.NUM_VARS)

        # dx = x_int - x_own
        vec[Star.X_INT] = 1
        vec[Star.X_OWN] = -1
        dx_min = self.star.minimize_vec(vec) @ vec
        dx_max = self.star.minimize_vec(-vec) @ vec

        qdx_min = floor(dx_min / pos_quantum)
        qdx_max = ceil(dx_max / pos_quantum)
        dx_qrange = (qdx_min, qdx_max)

        # dy = y_int - y_own
        vec = np.zeros(Star.NUM_VARS)
        #vec[Star.Y_INT] = 1 # Y_int is always 0
        vec[Star.Y_OWN] = -1

        dy_min = self.star.minimize_vec(vec) @ vec
        dy_max = self.star.minimize_vec(-vec) @ vec
        
        qdy_min = floor(dy_min / pos_quantum)
        qdy_max = ceil(dy_max / pos_quantum)
        dy_qrange = (qdy_min, qdy_max)

        return dx_qrange, dy_qrange

def is_init_qstate(qstate):
    """is this an initial qstate?

    returns True if any of the corners is inside the collision circle
    """

    rv = False
    qx, qy = qstate

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

class BackreachResult(TypedDict):
    counterexample: Optional[State]
    runtime: float
    num_popped: int
    unique_paths: int
    #params: Tuple[int, Tuple[int, int], Tuple[int, int], int, int, int]

def backreach_single(init_alpha_prev: int, x_own: Tuple[int, int], y_own: Tuple[int, int],
                     theta1: int, v_own: int, v_int: int, plot=False) -> Optional[BackreachResult]:
    """run backreachability from a single symbolic state"""

    global shared_num_counterexamples

    if shared_num_counterexamples.value > 0:
        return None

    start = time.perf_counter()

    box, a_mat, b_vec = init_to_constraints(x_own, y_own, v_own, v_int, theta1)

    init_star = Star(box, a_mat, b_vec)
    init_s = State(init_alpha_prev, theta1, v_own, v_int, init_star)

    #p = Plotter()
    #p.plot_star(init_s.star)

    work = [init_s]
    #colors = ['b', 'lime', 'r', 'g', 'magenta', 'cyan', 'pink']
    popped = 0

    #params = (init_alpha_prev, x_own, y_own, theta1, v_own, v_int)
    rv: BackreachResult = {'counterexample': None, 'runtime': np.inf,
                           'num_popped': 0, 'unique_paths': 0}
    deadends = set()

    plotter: Optional[Plotter] = None

    if plot:
        plotter = Plotter()
        plotter.plot_star(init_s.star, 'r')

    while work:

        if rv['counterexample'] is not None:
            print("break because of counterexample")
            break

        if popped % 100 == 0:
            if shared_num_counterexamples.value > 0:
                # don't need lock since we're just reading
                print("break because shared num_counterexamples > 0")
                break
        
        s = work.pop()
        #print(f"{popped}: {s.alpha_prev_list}")
        popped += 1

        predecessors = s.get_predecessors(plotter=plotter)

        for p in predecessors:
            work.append(p)
            
            if p.alpha_prev_list[-2] == 0 and p.alpha_prev_list[-1] == 0:
                rv['counterexample'] = deepcopy(p)

                with shared_num_counterexamples.get_lock():
                    shared_num_counterexamples.value += 1

                break

        if not predecessors:
            deadends.add(tuple(s.alpha_prev_list))

    #print(f"num popped: {popped}")
    #print(f"unique paths: {len(deadends)}")

    #for i, path in enumerate(deadends):
    #    print(f"{i+1}: {path}")

    diff = time.perf_counter() - start
    rv['runtime'] = diff
    rv['num_popped'] = popped
    rv['unique_paths'] = len(deadends)

    increment_progress()

    return rv

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
                for y_own_start in range(y_own_min, y_own_max):
                    y_own = (y_own_start, y_own_start + 1)
                    for x_own_start in range(x_own_min, x_own_max):
                        x_own = (x_own_start, x_own_start + 1)

                        if not is_init_qstate((x_own_start, y_own_start)):
                            continue

                        for qtheta1 in range(0, max_qtheta1):
                            params = (alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint)
                            params_list.append(params)

    return params_list

def run_all():
    """loop over all cases"""

    global global_total_num_cases
    global global_start_time

    # shared variable for coordination
    global_start_time = time.perf_counter()

    params_list = make_params()

    global_total_num_cases = len(params_list)
    diff = time.perf_counter() - global_start_time
    print(f"Made params for {global_total_num_cases} cases in {round(diff, 2)} secs")

    print("debug exit")
    exit(1)

    with multiprocessing.Pool() as pool:
        res_list = pool.starmap(backreach_single, params_list)
        max_runtime_result_params = (-np.inf, None, None)
        total_runtime = 0.0
        has_skipped_case = False
        counterexample = None

        for params, res in zip(params_list, res_list):
            if res is None:
                has_skipped_case = True
                continue

            if res['counterexample'] is not None:
                print("!!!!! counterexample!!!!")
                counterexample = res['counterexample']
                break

            t = res['runtime']
            total_runtime += t

            if t > max_runtime_result_params[0]:
                max_runtime_result_params = (t, res, params)

    if max_runtime_result_params[1] is not None and max_runtime_result_params[2] is not None:
        diff = max_runtime_result_params[0]
        res = max_runtime_result_params[1]
        alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint = max_runtime_result_params[2] 
        
        num_popped = res['num_popped']
        unique_paths = res['unique_paths']
        unsafe = res['counterexample'] is not None

        print(f"\nlongest runtime ({round(diff, 2)}) for:\nalpha_prev={alpha_prev}\nx_own={x_own}\n" + \
              f"y_own={y_own}\nqtheta1={qtheta1}\nq_vown={q_vown}\nq_vint={q_vint}")
        print(f'num_popped: {num_popped}, unique_paths: {unique_paths}, has_counterexample: {unsafe}')

    if counterexample is not None:
        print("\nCounterexample:")
        counterexample.print_replay_init()
        counterexample.print_replay_witness(plot=False)
        print("replay matches")

    if not has_skipped_case:
        diff = time.perf_counter() - global_start_time
        print(f"\nfinished enumeration ({global_total_num_cases} cases) in {to_time_str(diff)}")
        print(f"Avg runtime per case: {to_time_str(total_runtime / global_total_num_cases)}")
    else:
        print("\nIncomplete analysis; skipped some cases.")

def run_single():
    """test a single (difficult) case"""

    print("running single...")

    alpha_prev=3
    x_own=(0, 1)
    y_own=(1, 2)
    qtheta1=105
    q_vown=2
    q_vint=7
    plot = False

    Timers.tic('top')
    res = backreach_single(alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint, plot=plot)
    Timers.toc('top')
    Timers.print_stats()

    if res is not None:
        print(f"popped: {res['num_popped']}")
        print(f"unique_paths: {res['unique_paths']}")

    plt.show()

def main():
    """main entry point"""

    State.init_class()

    #run_single()
    run_all()

if __name__ == "__main__":
    main()
