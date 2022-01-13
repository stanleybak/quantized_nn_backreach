'''
Backreach using quantized inputs
'''

from typing import List, Tuple, Dict, TypedDict, Optional

import time
from copy import deepcopy
from math import floor, ceil
import traceback

import numpy as np
import matplotlib.pyplot as plt

from star import Star
from plotting import Plotter
from dubins import init_to_constraints, get_time_elapse_mat
from util import make_qstar, make_large_qstar, is_init_qx_qy, get_num_cores, get_tau_index
from networks import get_cmd

from timerutil import timed
from settings import Settings
from parallel import run_all_parallel, increment_index, shared_num_counterexamples, \
                     worker_had_counterexample, refine_indices, run_single_case, shared_num_timeouts

class State():
    """state of backreach container

    state is:
    alpha_prev (int) - previous command 
    """

    debug = False

    nn_update_rate = 1.0
    next_state_id = 0

    def __init__(self, alpha_prev: int, qtheta1: int, qv_own: int, qv_int: int, \
                 star: Star):

        assert isinstance(qtheta1, int)

        self.tau = 0
        self.qtheta1 = qtheta1
        self.qv_own = qv_own
        self.qv_int = qv_int      
        
        self.alpha_prev_list = [alpha_prev]
        
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
        print(f"tau_init = {self.tau}")

        domain_pt, range_pt, rad = self.star.get_witness(get_radius=True)
        print(f"# chebeshev center radius: {rad}")
        print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

    def print_replay_witness(self, plot=False):
        """print a step-by-step replay for the witness point"""

        s = self
        
        _, range_pt, rad = s.star.get_witness(get_radius=True)

        if rad < 1e-6:
            print(f"WARNING: radius was tiny ({rad}), skipping replay (may mismatch due to numerics)")
        else:
            p = Plotter()

            print()

            pt = range_pt.copy()

            q_theta1 = s.qtheta1
            s_copy = deepcopy(s)

            p.plot_star(s.star, color='r')
            mismatch = False

            pos_quantum = Settings.pos_q

            for i in range(len(s.alpha_prev_list) - 1):
                alpha_prev = s.alpha_prev_list[-(i+1)]
                expected_cmd = s.alpha_prev_list[-(i+2)]

                dx = floor((pt[Star.X_INT] - pt[Star.X_OWN]) / pos_quantum)
                dy = floor((0 - pt[Star.Y_OWN]) / pos_quantum)

                qstate = (dx, dy, q_theta1, s.qv_own, s.qv_int)

                tau_index = get_tau_index(s.tau)
                cmd_out = get_cmd(alpha_prev, tau_index, *qstate)
                print(f"({i+1}). alpha_prev={alpha_prev}, (tau, tau_index)={s.tau, tau_index} -> {cmd_out}")
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

                delta_q_theta = Settings.cmd_quantum_list[cmd_out]# * theta1_quantum
                q_theta1 += delta_q_theta

            if plot:
                plt.show()

            if mismatch:
                print("mismatch in replay... was the chebyshev center radius tiny?")
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
        delta_q_theta = Settings.cmd_quantum_list[cmd]

        if forward:
            self.qtheta1 += delta_q_theta
            self.tau += Settings.tau_dot
        else:
            self.qtheta1 -= delta_q_theta
            self.tau -= Settings.tau_dot

    @timed
    def get_predecessors(self, stdout=False):
        """get the valid predecessors of this star
        """

        dx_qrange, dy_qrange = self.get_dx_dy_qrange(stdout=stdout)

        # compute previous state
        self.backstep()

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
                    if is_init_qx_qy(qdx, qdy):
                        continue

                    tau_index = get_tau_index(self.tau)
                    out_cmd = get_cmd(prev_cmd, tau_index, qdx, qdy, *constants)

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
            
            # no splitting; all correct
            if all_right:
                prev_s = self.copy()
                prev_s.alpha_prev_list.append(prev_cmd)

                rv.append(prev_s)
                continue

            check_if_all_incorrect_infeasible = True

            if check_if_all_incorrect_infeasible:
                # still have a chance at all correct (no splitting): if all incorrect state are infeasible
                all_incorrect_infeasible = True

                for qstate in incorrect_qstates:
                    star = make_qstar(self.star, qstate)

                    if star.is_feasible() is not None:
                        all_incorrect_infeasible = False
                        break

                # no splitting; all incorrect states were infeasible
                if all_incorrect_infeasible:
                    prev_s = self.copy()
                    prev_s.alpha_prev_list.append(prev_cmd)

                    rv.append(prev_s)
                    continue

            split_all_single = False

            if split_all_single:
                # split along individual quantum boundaries
                for qstate in correct_qstates:
                    star = make_qstar(self.star, qstate)
                    domain_witness = star.is_feasible()

                    if domain_witness is not None:
                        prev_s = self.copy(star)
                        prev_s.alpha_prev_list.append(prev_cmd)

                        rv.append(prev_s)
            else:
                # split along multi-quantum boundaries, similar to run-length encoding
                
                while correct_qstates:
                    # pop top left
                    qdx, qdy = min(correct_qstates)
                    index = correct_qstates.index((qdx, qdy))
                    #single_star = feasible_correct_stars.pop(index)
                    correct_qstates.pop(index)

                    min_x = qdx
                    max_x = qdx + 1

                    # expand x
                    while True:
                        qstate = (max_x, qdy)

                        if qstate in correct_qstates:
                            correct_qstates.remove(qstate)
                            max_x += 1
                        else:
                            break

                    # expand y
                    min_y = qdy
                    max_y = qdy + 1
                    
                    while True:
                        all_in = True
                        
                        for x in range(min_x, max_x):
                            qstate = (x, max_y)
                            if qstate not in correct_qstates:
                                all_in = False
                                break

                        if all_in:
                            # expand y                            
                            for x in range(min_x, max_x):
                                qstate = (x, max_y)
                                correct_qstates.remove(qstate)

                            max_y += 1
                        else:
                            break

                    force_single_dx_dy = False

                    if force_single_dx_dy:
                        for x in range(min_x, max_x):
                            for y in range(min_y, max_y):
                                qstate = (x, y)
                                star = make_large_qstar(self.star, x, x + 1, y, y+1)

                                if star.is_feasible() is not None:
                                    prev_s = self.copy(star)
                                    prev_s.alpha_prev_list.append(prev_cmd)
                                    rv.append(prev_s)
                    else:
                        large_star = make_large_qstar(self.star, min_x, max_x, min_y, max_y)

                        if large_star.is_feasible() is not None:
                            prev_s = self.copy(large_star)
                            prev_s.alpha_prev_list.append(prev_cmd)
                            rv.append(prev_s)
        return rv

    @timed
    def get_dx_dy_qrange(self, stdout=False) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """get the quantized range for (dx, dy)"""

        pos_quantum = Settings.pos_q
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

class BackreachResult(TypedDict):
    counterexample: Optional[State]
    runtime: float
    num_popped: int
    unique_paths: int
    index: int
    params: Tuple[int, int, int, int, int, int]
    timeout: bool

def backreach_single(arg, parallel=True, plot=False) -> Optional[BackreachResult]:
    """run backreachability from a single symbolic state"""

    try:
        rv = backreach_single_unwrapped(arg, parallel=parallel, plot=plot)
    except:
        print("WARNING: Exception was raised!")
        traceback.print_exc()
        rv = None

    return rv

def backreach_single_unwrapped(arg, parallel=True, plot=False) -> Optional[BackreachResult]:
    """run backreachability from a single symbolic state"""

    if parallel:
        index, params = increment_index()
    else:
        assert arg is not None
        index = 0
        params = arg

    if index < 0: # exceeded max counterexamples
        return None

    init_alpha_prev, x_own, y_own, theta1, v_own, v_int = params

    start = time.perf_counter()

    box, a_mat, b_vec = init_to_constraints(x_own, y_own, v_own, v_int, theta1)

    init_star = Star(box, a_mat, b_vec)
    init_s = State(init_alpha_prev, theta1, v_own, v_int, init_star)

    work = [init_s]
    popped = 0

    rv: BackreachResult = {'counterexample': None, 'runtime': np.inf, 'params': params,
                           'num_popped': 0, 'unique_paths': 0, 'index': index, 'timeout': False}
    deadends = set()
    unique_prefixes = None

    plotter: Optional[Plotter] = None

    if plot:
        unique_prefixes = set()
        plotter = Plotter()
        plotter.plot_star(init_s.star, 'r')

    start = time.perf_counter()

    while work and rv['counterexample'] is None:
        s = work.pop()
        popped += 1

        if popped % 100 == 0 and Settings.max_counterexamples is not None and \
            shared_num_counterexamples.value > Settings.max_counterexamples:
            return None

        if time.perf_counter() - start > Settings.single_case_timeout:
            rv['timeout'] = True

            if parallel:
                with shared_num_timeouts.get_lock():
                    shared_num_timeouts.value += 1
            
            break

        if plotter is not None:
            tup = tuple(s.alpha_prev_list)
            if not tup in unique_prefixes:
                plotter.plot_star(s.star, 'k')
                unique_prefixes.add(tup)

        if not parallel and popped % 1000 == 0:
            lens = [len(w.alpha_prev_list) for w in work]
            max_len = max(lens)
            
            print(f"popped {popped}, unique_paths: {len(deadends)}, remaining_work: {len(work)}, max_len: {max_len}")

        predecessors = s.get_predecessors()

        for p in predecessors:
            work.append(p)
            
            if p.alpha_prev_list[-2] == 0 and p.alpha_prev_list[-1] == 0:
                # also check if > 20000 ft
                pt = p.star.get_witness()[1]

                dx = (pt[Star.X_INT] - pt[Star.X_OWN])
                dy = (0 - pt[Star.Y_OWN])
                
                if dx**2 + dy**2 > Settings.counterexample_start_dist**2:
                    rv['counterexample'] = deepcopy(p)

                    if parallel:
                        with shared_num_counterexamples.get_lock():
                            shared_num_counterexamples.value += 1
                            print(f"\nIndex {index} found counterexample. Count: {shared_num_counterexamples.value} ",
                                  end='', flush=True)

                    break

        if not predecessors:
            deadends.add(tuple(s.alpha_prev_list))

    diff = time.perf_counter() - start
    rv['runtime'] = diff
    rv['num_popped'] = popped
    rv['unique_paths'] = len(deadends)

    if rv['counterexample'] is not None and parallel:
        worker_had_counterexample(rv)

    return rv

def main():
    """main entry point"""

    Settings.init_cmd_quantum_list()

    # emacs hard-warp command: M+x fill-paragraph
    safe = run_all_parallel(backreach_single, 0)

    if safe:
        print("completed proof for tau_dot=0 case")

        safe = run_all_parallel(backreach_single, -1)
        print("completed proof for tau_dot=-1 case")
        print(f"final proven safe?: {safe}")
    else:
        print("not proven safe for for tau_dot = 0")

    #refine_indices(backreach_single, counterexample_indices)

if __name__ == "__main__":
    main()
