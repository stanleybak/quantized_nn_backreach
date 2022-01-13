"""
Settings for quantized backreach
"""

from typing import List
from math import pi, ceil, sqrt

# real? counter-example found with quanta: 10,10,0.1

class Settings:
    pos_q = 250
    vel_q = 0
    theta1_q_deg = 1.5 # should divide 1.5 degrees evenly
    
    range_vel_ownship = (200, 200) # full range: (100, 1200)                                                                         
    range_vel_intruder = (185, 185) # full range: (0, 1200)

    theta1_q = 2*pi / (360 / theta1_q_deg) # radians

    # how many theta1 quanta change for each command
    cmd_quantum_list: List[int] = [] # [0, 1, -1, 2, -2]

    # other settings
    single_case_timeout = 60 #15 * 60
    counterexample_start_dist = 60760 + ceil(sqrt(pos_q))


    # maximum counterexamples before starting refinement
    max_counterexamples = 128

    tau_dot = -1

    @classmethod
    def init_cmd_quantum_list(cls):
        '''init class variables'''

        theta1_quantum = cls.theta1_q
        
        q = 2*pi / (360 / 1.5)
        assert theta1_quantum * round(q/theta1_quantum) - q < 1e-6

        # initialize
        q = 2*pi / (360 / 1.5)
        cls.cmd_quantum_list = [0, round(q/theta1_quantum), -1 * round(q/theta1_quantum),
                                         2 * round(q/theta1_quantum), -2 * round(q/theta1_quantum)]

        assert Settings.tau_dot in [0, -1]
