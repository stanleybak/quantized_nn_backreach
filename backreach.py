'''
Backreach using quantized inputs
'''

from typing import List, Tuple, Dict

import time
from copy import deepcopy
from math import pi, floor, ceil

import numpy as np
import matplotlib.pyplot as plt

from star import Star
from plotting import Plotter
from dubins import init_to_constraints, get_time_elapse_mat
from util import quantize, make_qstar
from networks import get_cmd

from timerutil import timed, Timers
from settings import pos_quantum, vel_quantum, theta1_quantum

counter = 0

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
        self.qtheta1 = qtheta1
        self.qv_own = qv_own
        self.qv_int = qv_int      
        
        self.alpha_prev_list = [alpha_prev]

        if State.debug:
            self.qinput_star_list = []
        else:
            self.qinput_star_list = None
        
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
            for i in range(len(s.qinput_star_list) - 1, -1, -1):
                step += 1
                qinput, star = s.qinput_star_list[i]

                print(f"{step}. network {s.alpha_prev_list[i+1]} with qinput: {qinput} " + \
                      f"gave cmd {s.alpha_prev_list[i]}")

                p.plot_star(star, color='magenta')

            print()

        pq = pos_quantum
        pt = range_pt.copy()
        
        q_theta1 = s.qtheta1
        q_theta2 = 0

        s_copy = deepcopy(s)

        p.plot_star(s.star, color='r')
        mismatch = False

        for i in range(len(s.alpha_prev_list) - 1):
            net = s.alpha_prev_list[-(i+1)]
            expected_cmd = s.alpha_prev_list[-(i+2)]

            dx = quantize(pt[Star.X_INT] - pt[Star.X_OWN], pq)
            dy = quantize(0 - pt[Star.Y_OWN], pq)

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

            cmd_quantum_list = [0, 1, -1, 2, -2]
            delta_q_theta = cmd_quantum_list[cmd_out] * theta1_quantum
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
        cmd_quantum_list = [0, 1, -1, 2, -2]
        delta_q_theta = cmd_quantum_list[cmd]
        # note: this assume theta1 quantum is 1.5 degrees

        if forward:
            self.qtheta1 += delta_q_theta
        else:
            self.qtheta1 -= delta_q_theta

    @timed
    def get_predecessors(self, plotter=None, stdout=False):
        """get the valid predecessors of this star
        """

        global counter

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
            
            for qdx in range(dx_qrange[0], dx_qrange[1] + 1):
                for qdy in range(dy_qrange[0], dy_qrange[1] + 1):
                    qstate = (qdx, qdy)

                    out_cmd = get_cmd(prev_cmd, *qstate, *constants)

                    qstate_to_cmd[qstate] = out_cmd

                    if out_cmd == self.alpha_prev_list[-1]:
                        # command is correct
                        all_wrong = False
                    else:
                        # command is wrong
                        all_right = False

            if all_wrong:
                continue
            
            # for the current prev_command, was either all correct or all wrong?
            if all_right:
                prev_s = self.copy()
                prev_s.alpha_prev_list.append(prev_cmd)
                rv.append(prev_s)
                continue

            # still have a chance at all correct: if incorrect state are infeasible
            all_incorrect_infeasible = True
            
            for qdx in range(dx_qrange[0], dx_qrange[1] + 1):
                for qdy in range(dy_qrange[0], dy_qrange[1] + 1):
                    qstate = (qdx, qdy)
                    out_cmd = qstate_to_cmd.get(qstate)

                    if out_cmd != self.alpha_prev_list[-1]:
                        # if incorrect commands
                        star = make_qstar(self.star, qstate)

                        if star.is_feasible():
                            all_incorrect_infeasible = False
                            break

                if not all_incorrect_infeasible:
                    break

            if all_incorrect_infeasible:
                prev_s = self.copy()
                prev_s.alpha_prev_list.append(prev_cmd)
                rv.append(prev_s)
                continue

            # ok, do splitting if command is correct
            correct_qstates = []
            incorrect_qstates = []
            
            for qdx in range(dx_qrange[0], dx_qrange[1] + 1):
                for qdy in range(dy_qrange[0], dy_qrange[1] + 1):
                    qstate = (qdx, qdy)
                    star = make_qstar(self.star, qstate)

                    if star.is_feasible():
                        out_cmd = qstate_to_cmd.get(qstate)

                        if out_cmd == self.alpha_prev_list[-1]:
                            correct_qstates.append(qstate)
                            
                            prev_s = self.copy(star)
                            prev_s.alpha_prev_list.append(prev_cmd)
                            rv.append(prev_s)
                        else:
                            incorrect_qstates.append(qstate)

            # split along incorrect qstates
            # each element is a_mat, rhs_list, states
            groups: List[Tuple[List[List[float]], List[float], List[Tuple[int, int]]]] = [([], [], correct_qstates)]

            # dx = x_int - x_own
            dx_row = [0.0] * Star.NUM_VARS
            dx_row[Star.X_INT] = 1
            dx_row[Star.X_OWN] = -1

            neg_dx_row = [-x for x in dx_row]

            # dy = y_int - y_own
            dy_row = [0.0] * Star.NUM_VARS
            #dy_row[Star.Y_INT] = 1 # always 0
            dy_row[Star.Y_OWN] = -1

            neg_dy_row = [-y for y in dy_row]

            for ix, iy in incorrect_qstates:
                new_groups: List[Tuple[List[List[float]], List[float], List[Tuple[int, int]]]] = []

                for mat, rhs, qstates in groups:
                    case1 = []
                    case2 = []
                    case3 = []
                    case4 = []

                    for qstate in qstates:
                        x, y = qstate

                        if x < ix:
                            case1.append(qstate)
                        elif x > ix:
                            case2.append(qstate)
                        elif x == ix and y > iy:
                            case3.append(qstate)
                        else:
                            assert x == ix and y < iy
                            case4.append(qstate)

                    if case1:
                        new_mat = mat.copy()
                        new_rhs = rhs.copy()

                        # add x < ix constraint
                        new_mat.append(dx_row)
                        new_rhs.append(ix * pos_quantum)

                        tup = (new_mat, new_rhs, case1)
                        new_groups.append(tup)

                    if case2:
                        new_mat = mat.copy()
                        new_rhs = rhs.copy()

                        # add x > ix constraint
                        new_mat.append(neg_dx_row)
                        new_rhs.append(-((ix+1) * pos_quantum))

                        tup = (new_mat, new_rhs, case2)
                        new_groups.append(tup)

                    if case3:
                        new_mat = mat.copy()
                        new_rhs = rhs.copy()

                        # add x == ix and y > iy
                        new_mat.append(dx_row)
                        new_rhs.append((ix + 1) * pos_quantum)

                        new_mat.append(neg_dx_row)
                        new_rhs.append(-ix * pos_quantum)

                        new_mat.append(neg_dy_row)
                        new_rhs.append(-((iy + 1) * pos_quantum))

                        tup = (new_mat, new_rhs, case3)
                        new_groups.append(tup)

                    if case4:
                        new_mat = mat.copy()
                        new_rhs = rhs.copy()

                        # add x == ix and y < iy
                        new_mat.append(dx_row)
                        new_rhs.append((ix + 1) * pos_quantum)

                        new_mat.append(neg_dx_row)
                        new_rhs.append(-ix * pos_quantum)

                        new_mat.append(dy_row)
                        new_rhs.append(iy * pos_quantum)

                        tup = (new_mat, new_rhs, case4)
                        new_groups.append(tup)

                # assign for next iteration
                groups = new_groups

            #print(f"correct ({len(correct_qstates)}): {correct_qstates}")
            #print(f"incorrect ({len(incorrect_qstates)}): {incorrect_qstates}")
            #print(f"num groups: {len(groups)}")

            for mat, rhs, _ in groups:
                star = deepcopy(self.star)
                for row, val in zip(mat, rhs):
                    star.add_dense_row(row, val)

                # star should be feasible?!?
                assert star.is_feasible()
                prev_s = self.copy(star)

                prev_s.alpha_prev_list.append(prev_cmd)
                rv.append(prev_s)

            counter += 1
            
            if counter >= np.inf:
                p = Plotter()
                p.plot_star(self.star, 'k')

                for qdx in range(dx_qrange[0], dx_qrange[1] + 1):
                    for qdy in range(dy_qrange[0], dy_qrange[1] + 1):
                        qstate = (qdx, qdy)
                        star = make_qstar(self.star, qstate)

                        if star.is_feasible():
                            out_cmd = qstate_to_cmd.get(qstate)

                            if out_cmd != self.alpha_prev_list[-1]:
                                p.plot_star(star, 'r', zorder=2)

                for i, (mat, rhs, _) in enumerate(groups):
                    star = deepcopy(self.star)
                    for row, val in zip(mat, rhs):
                        star.add_dense_row(row, val)

                    # star should be feasible?!?
                    assert star.is_feasible()

                    color = ['g', 'b'][i % 2]
                    p.plot_star(star, color, zorder=2)
                
                plt.show()
                exit(1)
            
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

def backreach_single(init_alpha_prev: int, x_own: Tuple[int, int], y_own: Tuple[int, int],
                     theta1: int, v_own: int, v_int: int, target_len=None, plot=False):
    """run backreachability from a single symbolic state"""

    #q_v_int = (v_int[0] + v_int[1]) / 2
    #q_v_own = (v_own[0] + v_own[1]) / 2
    #q_theta1 = (theta1_own[0] + theta1_own[1]) / 2

    box, a_mat, b_vec = init_to_constraints(x_own, y_own, v_own, v_int, theta1)

    init_star = Star(box, a_mat, b_vec)
    init_s = State(init_alpha_prev, theta1, v_own, v_int, init_star)

    #p = Plotter()
    #p.plot_star(init_s.star)

    work = [init_s]
    #colors = ['b', 'lime', 'r', 'g', 'magenta', 'cyan', 'pink']
    popped = 0

    rv = {'counterexample': None, 'longest alpha_prev_list state': init_s}
    deadends = set()

    if plot:
        plotter = Plotter()
        plotter.plot_star(init_s.star, 'r')
    else:
        plotter = None

    while work:

        if rv['counterexample'] is not None:
            print("break because of counterexample")
            break

        if target_len is not None and len(rv['longest alpha_prev_list state'].alpha_prev_list) >= target_len:
            print(f"break because of alpha_prev_list len >= target({target_len})")
            break
        
        s = work.pop()
        #print(f"{popped}: {s.alpha_prev_list}")
        popped += 1

        predecessors = s.get_predecessors(plotter=plotter)

        for p in predecessors:
            work.append(p)
            
            if p.alpha_prev_list[-2] == 0 and p.alpha_prev_list[-1] == 0:
                rv['counterexample'] = deepcopy(p)

            if len(p.alpha_prev_list) > len(rv['longest alpha_prev_list state'].alpha_prev_list):
                rv['longest alpha_prev_list state'] = deepcopy(p)

        if not predecessors:
            deadends.add(tuple(s.alpha_prev_list))

    print(f"num popped: {popped}")
    print(f"unique paths: {len(deadends)}")

    #for i, path in enumerate(deadends):
    #    print(f"{i+1}: {path}")

    return rv

def run_all():
    """loop over all cases"""
    
    alpha_prev = 4

    x_own = (round(-500 / pos_quantum), round(-400 / pos_quantum))
    y_own = (round(-100 / pos_quantum), round(0 / pos_quantum))

    done = False
    s = None
    target_len = 30 #None

    max_qtheta1 = round(2*pi / theta1_quantum)
    
    for qtheta1 in range(0, max_qtheta1):
        
        for q_vown in range(round(100/vel_quantum), round(1200/vel_quantum)):
        
            print(f"\n{round(qtheta1/max_qtheta1, 2)}%-{q_vown*vel_quantum}/1200:", end='')

            for q_vint in range(0, round(1200/vel_quantum)):
                print(".", flush=True, end='')

                start = time.perf_counter()
                res = backreach_single(alpha_prev, x_own, y_own, qtheta1, q_vown, q_vint, target_len=target_len)
                diff = time.perf_counter() - start

                if diff > 1.0:
                    print(f"\nlong runtime ({round(diff, 2)}) for:\nalpha_prev={alpha_prev}\nx_own={x_own}\n" + \
                          f"y_own={y_own}\nqtheta1={qtheta1}\nq_vown={q_vown}\nq_vint={q_vint}")

                if len(res['longest alpha_prev_list state'].alpha_prev_list) >= target_len:
                    print("found target len")
                    s = res['longest alpha_prev_list state']
                    done = True
                    break

                if res['counterexample'] is not None:
                    print("!!!!! counterexample!!!!")
                    s = res['counterexample']
                    done = True
                    break

            if done:
                break

        if done:
            break

    if s is not None:
        print(f"longest alpha_prev_list was {len(s.alpha_prev_list)}: {s}\n")
                            
        s.print_replay_init()

        s.print_replay_witness(plot=True)

    if not done:
        print("\nfinished enumeration")

def run_single():
    """test a single (difficult)case"""

    print("running single...")

    alpha_prev=0
    
    x_own = (round(-500 / pos_quantum), round(-400 / pos_quantum))
    y_own = (round(-100 / pos_quantum), round(0 / pos_quantum))
    
    theta1_own = round(0.02617993877991494 / theta1_quantum)
    v_own = round(800 / vel_quantum)
    v_int = round(1100 / vel_quantum)

    Timers.tic('top')
    backreach_single(alpha_prev, x_own, y_own, theta1_own, v_own, v_int, plot=False)
    Timers.toc('top')
    Timers.print_stats()

    plt.show()

def main():
    """main entry point"""

    run_single()
    #run_all()

if __name__ == "__main__":
    main()
