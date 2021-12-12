'''
Backreach using quantized inputs
'''

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from star import Star
from plotting import Plotter
from dubins import init_to_constraints, get_time_elapse_mat

class State():
    """state of backreach container

    state is:
    alpha_prev (int) - previous command 
    
    """

    nn_update_rate = 1.0

    def __init__(self, alpha_prev:int, box_bounds, a_mat, b_vec):
        self.alpha_prev = alpha_prev
        self.star = Star(box_bounds, a_mat, b_vec)

    def backstep(self):
        # step backwards according to alpha_prev

        mat = get_time_elapse_mat(self.alpha_prev, -1.0)

        self.star.a_mat = self.star.a_mat @ mat
        self.star.b_vec = self.star.b_vec @ mat
       
        
def main():
    """main entry point"""

    x_own = (450, 500)
    y_own = (0, 50)
    v_int = (400, 450)
    alpha_prev = 4

    psi_own = (1*2*pi / 16, 2*2*pi / 16)
    v_own = (600, 650)

    box, a_mat, b_vec = init_to_constraints(v_own, v_int, x_own, y_own, psi_own)

    s = State(alpha_prev, box, a_mat, b_vec)

    p = Plotter()
    p.plot_state(s)

    # compute previous state
    s.backstep()
    p.plot_state(s, 'b-')
    
    plt.show()

if __name__ == "__main__":
    main()
