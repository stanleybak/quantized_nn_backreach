"""
Settings for quantized backreach
"""

from typing import List
from math import pi

# real? counter-example found with quanta: 10,10,0.1

class Quanta:
    pos = 250 #250
    vel = 50 # 50
    theta1_deg = 1.5 #1.5 # should divide 1.5 degrees evenly

    theta1 = 2*pi / (360 / theta1_deg)

    # how many theta1 quanta change for each command
    cmd_quantum_list: List[int] = [] # [0, 1, -1, 2, -2]

    # other settings
    single_case_timeout = 60 #15 * 60
    counterexample_start_dist = 20000

    @classmethod
    def init_cmd_quantum_list(cls):
        '''init class variables'''

        theta1_quantum = cls.theta1
        
        q = 2*pi / (360 / 1.5)
        assert theta1_quantum * round(q/theta1_quantum) - q < 1e-6

        # initialize
        q = 2*pi / (360 / 1.5)
        cls.cmd_quantum_list = [0, round(q/theta1_quantum), -1 * round(q/theta1_quantum),
                                         2 * round(q/theta1_quantum), -2 * round(q/theta1_quantum)]
