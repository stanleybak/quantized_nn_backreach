"""
Settings for quantized backreach
"""

from math import pi

pos_quantum = 100
vel_quantum = 100
theta1_quantum = 2*pi / (360 / 1.5) # every 1.5 degrees

quantization = (pos_quantum, vel_quantum, theta1_quantum)
