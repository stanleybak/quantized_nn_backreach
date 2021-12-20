'''
Backreach using quantized inputs
'''

from typing import List, Tuple

from copy import deepcopy
from math import pi, atan2, sin, cos

import numpy as np
import matplotlib.pyplot as plt

from star import Star
from plotting import Plotter
from dubins import init_to_constraints, get_time_elapse_mat
from util import quantize, make_qstar
from networks import qstate_cmd

class State():
    """state of backreach container

    state is:
    alpha_prev (int) - previous command 
    """

    nn_update_rate = 1.0

    pos_quantum = 100
    theta1_quantum = 2*pi / (360 / 1.5) # every 1.5 degrees

    next_state_id = 0

    def __init__(self, alpha_prev: int, q_theta1: float, q_v_own: float, q_v_int: float, star: Star):
        self.q_theta1 = q_theta1
        self.alpha_prev_list = [alpha_prev]
        self.qinput_list = []
        
        self.q_v_own = q_v_own
        self.q_v_int = q_v_int      
        
        self.star = star
        
        self.state_id = -1
        self.assign_state_id()

    def __str__(self):
        return f"State(id={self.state_id} with alpha_prev_list = {self.alpha_prev_list})"

    def print_replay_init(self):
        """print initialization states for replay"""

        print(f"alpha_prev_list = {self.alpha_prev_list}")
        print(f"q_theta1 = {self.q_theta1}")
        print(f"q_v_int = {self.q_v_int}")
        print(f"q_v_own = {self.q_v_own}")

    def assign_state_id(self):
        """assign and increment state_id"""

        self.state_id = State.next_state_id
        State.next_state_id += 1

    def backstep(self, forward=False, forward_alpha_prev=-1):
        """step backwards according to alpha_prev"""

        if forward:
            cmd = forward_alpha_prev
        else:
            cmd = self.alpha_prev_list[-1]

        assert 0 <= cmd <= 4

        mat = get_time_elapse_mat(cmd, -1.0 if not forward else 1.0)

        self.star.a_mat = mat @ self.star.a_mat
        self.star.b_vec = mat @ self.star.b_vec

        # clear, weak left, weak right, strong left, strong right
        cmd_quantum_list = [0, 1, -1, 2, -2]
        delta_q_theta = cmd_quantum_list[cmd] * State.theta1_quantum

        if forward:
            self.q_theta1 += delta_q_theta
        else:
            self.q_theta1 -= delta_q_theta

    def get_qstates_qstars(self, stdout=False) -> List[Tuple[Tuple[float, float, float, float, float, float], Star]]:
        """get the discretized states used as inputs to the nn

        returns a list of quantized states: (dx, dy, theta1, theta2, q_v_own, q_v_int)
            with their associated feasible Star
        """

        rv: List[Tuple[Tuple[float, float, float, float, float, float], Star]] = []
        limits = self.get_discretized_ranges(stdout=stdout)
        tol = 1e-6

        pq = State.pos_quantum

        for dx in np.arange(quantize(limits[0][0], pq), quantize(limits[0][1], pq) + tol, pq):
            for dy in np.arange(quantize(limits[1][0], pq), quantize(limits[1][1], pq) + tol, pq):
                q_theta2 = 0 # intruder is always moving right
                
                qstate = (dx, dy, self.q_theta1, q_theta2, self.q_v_own, self.q_v_int)

                qstar = make_qstar(self.star, qstate, pq)

                if qstar.is_feasible():
                    rv.append((qstate, qstar))

        assert rv, "quantized boxes was empty?"

        return rv
    
    def get_discretized_ranges(self, stdout=False) -> List[Tuple[float, float]]:
        """get the ranges for (dx, dy)"""

        rv = []

        zeros = np.zeros(Star.NUM_VARS)
        vec = zeros.copy()

        # dx = x_int - x_own
        vec[Star.X_INT] = 1
        vec[Star.X_OWN] = -1
        dx_min = self.star.minimize_vec(vec) @ vec
        dx_max = self.star.minimize_vec(-vec) @ vec
        rv.append((dx_min, dx_max))

        # dy = y_int - y_own
        vec = zeros.copy()
        #vec[Star.Y_INT] = 1 # Y_int is always 0
        vec[Star.Y_OWN] = -1

        if stdout:
            print(f"opt vec: {vec}")

            pt = self.star.minimize_vec(vec)
            print(f"min pt: {pt}, min val: {pt @ vec}")

            pt = self.star.minimize_vec(-vec)
            print(f"ax pt: {pt}, max val: {pt @ vec}")

        dy_min = self.star.minimize_vec(vec) @ vec
        dy_max = self.star.minimize_vec(-vec) @ vec
        rv.append((dy_min, dy_max))

        if stdout:
            print(f"dy range: {rv[-1]}")

        return rv
    

def backreach_single(init_alpha_prev, x_own, y_own, theta1_own, v_own, v_int, target_len=None):
    """run backreachability from a single symbolic state"""

    q_v_int = (v_int[0] + v_int[1]) / 2
    q_v_own = (v_own[0] + v_own[1]) / 2
    q_theta1 = (theta1_own[0] + theta1_own[1]) / 2

    box, a_mat, b_vec = init_to_constraints(v_own, v_int, x_own, y_own, theta1_own)

    init_star = Star(box, a_mat, b_vec)
    init_s = State(init_alpha_prev, q_theta1, q_v_own, q_v_int, init_star)

    #p = Plotter()
    #p.plot_star(init_s.star)

    work = [init_s]
    #colors = ['b', 'lime', 'r', 'g', 'magenta', 'cyan', 'pink']
    popped = 0

    rv = {'counterexample': None, 'longest alpha_prev_list state': init_s}

    while work:

        if rv['counterexample'] is not None:
            print("break because of counterexample")
            break

        if target_len is not None and len(rv['longest alpha_prev_list state'].alpha_prev_list) >= target_len:
            print(f"break because of alpha_prev_list len >= target({target_len})")
            break
        
        s = work.pop()
        
        # compute previous state
        s.backstep()
        #p.plot_star(s.star, colors[popped % len(colors)])
        popped += 1

        q_list = s.get_qstates_qstars()
        print(f"------- popped {popped} -----------")
        print(f"popped state {s} num quantized states: {len(q_list)}")

        for alpha_prev in range(5):
            print(f"using alpha_prev network {alpha_prev}")

            all_correct = True
            correct_qstars_qinputs = []

            for qstate, qstar in q_list:
                cmd, qinput = qstate_cmd(alpha_prev, qstate)

                if cmd == s.alpha_prev_list[-1]:
                    correct_qstars_qinputs.append((qstar, qinput))
                else:
                    all_correct = False

            print(f"alpha_prev: {alpha_prev}, all_correct: {all_correct}, num_correct: {len(correct_qstars_qinputs)}")

            if all_correct:                
                # if all correct, propagate the entire set without splitting
                next_s = deepcopy(s)
                next_s.assign_state_id()
                next_s.alpha_prev_list.append(alpha_prev)
                work.append(next_s)

                qinputs = tuple(qi for _, qi in correct_qstars_qinputs)
                next_s.qinput_list.append(qinputs)

                print(f"appending to work, work len: {len(work)}")

                if next_s.alpha_prev_list[-2] == 0 and alpha_prev == 0:
                    # CoC network predicts CoC... valid start state!
                    rv['counterexample'] = next_s

                if len(next_s.alpha_prev_list) > len(rv['longest alpha_prev_list state'].alpha_prev_list):
                    rv['longest alpha_prev_list state'] = next_s

                print(f"details of predecessor state {next_s}")
                domain_pt, range_pt = next_s.star.get_witness()
                print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")
            else:
                state_strs = []
                
                for qstar, qinput in correct_qstars_qinputs:
                    next_s = deepcopy(s)
                    next_s.assign_state_id()
                    next_s.alpha_prev_list.append(alpha_prev)
                    next_s.star = qstar
                    work.append(next_s)

                    next_s.qinput_list.append(qinput)

                    if next_s.alpha_prev_list[-2] == 0 and alpha_prev == 0:
                        # CoC network predicts CoC... valid start state!
                        rv['counterexample'] = next_s

                    if len(next_s.alpha_prev_list) > len(rv['longest alpha_prev_list state'].alpha_prev_list):
                        rv['longest alpha_prev_list state'] = next_s

                    state_strs.append(str(next_s))

                    print(f"details of predecessor state {next_s}")
                    domain_pt, range_pt = next_s.star.get_witness()
                    print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

            if len(correct_qstars_qinputs) == 0:
                print(f"no predecessors of state {s}")

    #plt.tight_layout()
    #plt.show()
    print(f"work len: {len(work)}")
    
    return rv

def main():
    """main entry point"""
    
    delta_v = 100

    alpha_prev = 4

    x_own = (-500, -400)
    y_own = (-100, 0)

    #psi_own = (1*2*pi / 16, 2*2*pi / 16)

    ep = 1e-6
    done = False
    s = None
    target_len = 20
    
    for theta1_own_min in np.arange(0, 2*pi - ep, State.theta1_quantum):
        theta1_own = (theta1_own_min, theta1_own_min + State.theta1_quantum)
        
        for v_own_min in range(100, 1200, delta_v):

            for v_int_min in range(0, 1200, delta_v):

                v_own = (v_own_min, v_own_min + delta_v)
                v_int = (v_int_min, v_int_min + delta_v)

                res = backreach_single(alpha_prev, x_own, y_own, theta1_own, v_own, v_int, target_len=target_len)

                assert res['counterexample'] is None
                s = res['longest alpha_prev_list state']

                if len(s.alpha_prev_list) >= target_len:
                    done = True
                    break

            if done:
                break

        if done:
            break

    if s is not None:
        print(f"\nlongest alpha_prev_list was {len(s.alpha_prev_list)}: {s}")
                            
        s.print_replay_init()

        domain_pt, range_pt = s.star.get_witness()
        print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

        print()
        step = 0
        for i in range(len(s.qinput_list) - 1, -1, -1):
            step += 1
            print(f"{step}. network {s.alpha_prev_list[i+1]} with qinput: {s.qinput_list[i]} " + \
                  f"gave cmd {s.alpha_prev_list[i]}")

        print()
        alpha_prev = s.alpha_prev_list[-1]

        pq = State.pos_quantum
        pt = range_pt.copy()
        
        q_theta1 = s.q_theta1
        q_theta2 = 0

        p = Plotter()
        s_copy = deepcopy(s)

        #star_temp = deepcopy(s.star)
        #star_temp.a_mat = np.identity(6)
        
        #star_temp.a_mat = np.identity(star_temp.dims)
        #star_temp.b_vec = np.zeros(star_temp.dims)

        #p.plot_star(star_temp)

        #plt.show()
        #print("debug exit")
        #exit(1)

        p.plot_star(s.star)

        for i in range(len(s.alpha_prev_list) - 1):
            net = s.alpha_prev_list[-(i+1)]
            expected_cmd = s.alpha_prev_list[-(i+2)]

            dx = quantize(pt[Star.X_INT] - pt[Star.X_OWN], pq)
            dy = quantize(0 - pt[Star.Y_OWN], pq)

            qstate = (dx, dy, q_theta1, q_theta2, s.q_v_own, s.q_v_int)
            
            cmd_out, qinput = qstate_cmd(net, qstate)
            print(f"({i+1}). network {net} with qinput: {qinput} -> {cmd_out}")
            print(f"state: {list(pt)}")
            print(f"qstate: {qstate}")

            s_copy.backstep(forward=True, forward_alpha_prev=cmd_out)
            p.plot_star(s_copy.star)

            assert cmd_out == expected_cmd, f"got cmd {cmd_out}, expected cmd {expected_cmd}"
            mat = get_time_elapse_mat(cmd_out, 1.0)
            pt = mat @ pt

            cmd_quantum_list = [0, 1, -1, 2, -2]
            delta_q_theta = cmd_quantum_list[cmd_out] * State.theta1_quantum
            q_theta1 += delta_q_theta

            if i == 5:
                break

        plt.show()

    if not done:
        print("\nfinished enumeration")

if __name__ == "__main__":
    main()
