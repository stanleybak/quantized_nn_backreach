'''
Backreach using quantized inputs
'''

from typing import List, Tuple

from copy import deepcopy
from math import pi

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
    vel_quantum = 100

    next_state_id = 0

    def __init__(self, alpha_prev:int, q_v_own:float, q_v_int:float, star:Star):
        self.alpha_prev_list = [alpha_prev]
        self.q_v_own = q_v_own
        self.q_v_int = q_v_int
        
        self.star = star
        
        self.state_id = -1
        self.assign_state_id()

    def __str__(self):
        return f"State(id={self.state_id} with alpha_prev_list = {self.alpha_prev_list})"

    def assign_state_id(self):
        """assign and increment state_id"""

        self.state_id = State.next_state_id
        State.next_state_id += 1

    def backstep(self):
        """step backwards according to alpha_prev"""

        mat = get_time_elapse_mat(self.alpha_prev_list[-1], -1.0)

        self.star.a_mat = self.star.a_mat @ mat
        self.star.b_vec = self.star.b_vec @ mat

    def get_qstates_qstars(self) -> List[Tuple[Tuple[float, float, float, float, float, float, float], Star]]:
        """get the discretized states used as inputs to the nn

        returns a list of quantized states: (dx, dy, dvx_own, dvy_own, dvx_int, q_v_own, q_v_int)
            with their associated feasible Star
        """

        rv: List[Tuple[Tuple[float, float, float, float, float, float, float], Star]] = []
        limits = self.get_discretized_ranges()
        tol = 1e-6

        pq = State.pos_quantum
        vq = State.vel_quantum

        for dx in np.arange(quantize(limits[0][0], pq), quantize(limits[0][1], pq) + tol, pq):
            for dy in np.arange(quantize(limits[1][0], pq), quantize(limits[1][1], pq) + tol, pq):
                for vxo in np.arange(quantize(limits[2][0], vq), quantize(limits[2][1], vq) + tol, vq):
                    for vyo in np.arange(quantize(limits[3][0], vq), quantize(limits[3][1], vq) + tol, vq):
                        for vxi in np.arange(quantize(limits[4][0], vq), quantize(limits[4][1], vq) + tol, vq):
                            qstate = (dx, dy, vxo, vyo, vxi, self.q_v_own, self.q_v_int)

                            # todo: maybe can filter out qstate if q_v_own or q_v_int does not correspond to the dx/dy??

                            qstar = make_qstar(self.star, qstate, pq, vq)

                            if qstar.is_feasible():
                                rv.append((qstate, qstar))

        assert rv, "quantized boxes was empty?"

        return rv
    
    def get_discretized_ranges(self) -> List[Tuple[float, float]]:
        """get the ranges for (dx, dy, vx_own, vy_own, vx_int)"""

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

        dy_min = self.star.minimize_vec(vec) @ vec
        dy_max = self.star.minimize_vec(-vec) @ vec
        rv.append((dy_min, dy_max))

        # vx_own
        vec = zeros.copy()
        vec[Star.VX_OWN] = 1

        vxo_min = self.star.minimize_vec(vec) @ vec
        vxo_max = self.star.minimize_vec(-vec) @ vec
        rv.append((vxo_min, vxo_max))

        # vy_own
        vec = zeros.copy()
        vec[Star.VY_OWN] = 1

        vyo_min = self.star.minimize_vec(vec) @ vec
        vyo_max = self.star.minimize_vec(-vec) @ vec
        rv.append((vyo_min, vyo_max))

        # vx_int
        vec = zeros.copy()
        vec[Star.VX_INT] = 1

        vxi_min = self.star.minimize_vec(vec) @ vec
        vxi_max = self.star.minimize_vec(-vec) @ vec
        rv.append((vxi_min, vxi_max))

        return rv
        
def main():
    """main entry point"""

    print("LATEST IDEA: just have a theta discretization for ownship and intruder, not for vx, vy")
    print("you can instersect star in specific discretization angles, since it's pizza slices on vx/vy plane")
    exit(1)

    alpha_prev = 4

    x_own = (-500, -450)
    y_own = (-50, 0)
    #psi_own = (1*2*pi / 16, 2*2*pi / 16)
    psi_own = (0*2*pi / 32, 1*2*pi / 32)
        
    v_int = (425, 475)
    v_own = (625, 675)

    assert v_int[1] - v_int[0] == 50 and v_int[0] % 50 == 25
    assert v_own[1] - v_own[0] == 50 and v_own[0] % 50 == 25
    q_v_int = (v_int[0] + v_int[1]) / 2
    q_v_own = (v_own[0] + v_own[1]) / 2

    box, a_mat, b_vec = init_to_constraints(v_own, v_int, x_own, y_own, psi_own)

    init_star = Star(box, a_mat, b_vec)
    init_s = State(alpha_prev, q_v_own, q_v_int, init_star)

    p = Plotter()
    p.plot_star(init_s.star)

    work = [init_s]
    colors = ['b', 'lime', 'r', 'g', 'magenta', 'cyan', 'pink']
    popped = 0

    while work:
        s = work.pop()
        
        # compute previous state
        s.backstep()
        p.plot_star(s.star, colors[popped % len(colors)])
        popped += 1

        q_list = s.get_qstates_qstars()
        print(f"popped state {s} num quantized states: {len(q_list)}")

        #p.plot_quantization(l)
        no_predecessors = True

        for alpha_prev in range(5):
            cmds = [qstate_cmd(alpha_prev, qstate) for qstate, _ in q_list]

            all_correct = all(cmd == s.alpha_prev_list[-1] for cmd in cmds[1:])
            num_correct = 0

            if all_correct:                
                # if all correct, propagate the entire set without splitting
                next_s = deepcopy(s)
                next_s.assign_state_id()
                next_s.alpha_prev_list.append(alpha_prev)
                work.append(next_s)

                print(f"alpha_prev: {alpha_prev}, all_correct: {all_correct} - {next_s}")

                domain_pt, range_pt = next_s.star.get_center_witness()
                print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")
            else:
                state_strs = []
                    
                for (qstate, qstar), cmd in zip(q_list, cmds):
                    if cmd == s.alpha_prev_list[-1]:
                        num_correct += 1

                        next_s = deepcopy(s)
                        next_s.assign_state_id()
                        next_s.alpha_prev_list.append(alpha_prev)
                        next_s.star = qstar
                        work.append(next_s)

                        state_strs.append(str(next_s))

                        print(f"details of state {next_s}")
                        domain_pt, range_pt = s.star.get_center_witness()
                        print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

                        qstate_cmd(alpha_prev, qstate, stdout=True)
                        
                        exit(1)

                print(f"alpha_prev {alpha_prev}: num correct: {num_correct}/{len(cmds)} - {state_strs}")

            if not all_correct and num_correct == 0:
                print(f"no predecessors of state {s}")
                domain_pt, range_pt = s.star.get_center_witness()
                print(f"end = np.{repr(domain_pt)}\nstart = np.{repr(range_pt)}")

        if popped == 1:
            print(f"work queue size: {len(work)}")
            plt.tight_layout()
            #plt.show()
            break

    print(f"q_v_int = {q_v_int}")
    print(f"q_v_own = {q_v_own}")

if __name__ == "__main__":
    main()
