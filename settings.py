"""
Settings for quantized backreach
"""

from math import pi

# real? counter-example found with quanta: 10,10,0.1
pos_quantum = 500
vel_quantum = 100
theta1_deg = 1.5 # should divide 1.5 degrees evenly

theta1_quantum = 2*pi / (360 / theta1_deg) 
