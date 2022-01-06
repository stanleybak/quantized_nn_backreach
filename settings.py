"""
Settings for quantized backreach
"""

from math import pi

# real? counter-example found with quanta: 10,10,0.1
pos_quantum = 250 #250
vel_quantum = 50 # 50
theta1_deg = 1.5 # should divide 1.5 degrees evenly

theta1_quantum = 2*pi / (360 / theta1_deg) 

# in a few minutes of computation time,
# with pos_quantum = 500, a 84-step command sequence was found where
# the aircraft was initially turning weak-left and clear-of-conflict
# and then switched to weak and strong right before the collision State
# was reached

