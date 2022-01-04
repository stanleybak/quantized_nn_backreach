"""
Settings for quantized backreach
"""

from math import pi

# real? counter-example found with quanta: 10,10,0.1
pos_quantum = 250
vel_quantum = 100
theta1_deg = 1.5 # should divide 1.5 degrees evenly

theta1_quantum = 2*pi / (360 / theta1_deg) 

# in a few minutes of computation time,
# with pos_quantum = 500, a 84-step command sequence was found where
# the aircraft was initially turning weak-left and clear-of-conflict
# and then switched to weak and strong right before the collision State
# was reached

assert -500 % pos_quantum < 1e-6
assert 500 % pos_quantum < 1e-6

x_own_min = round(-500 / pos_quantum)
x_own_max = round(500 / pos_quantum)

y_own_min = round(-500 / pos_quantum)
y_own_max = round(500 / pos_quantum)

max_qtheta1 = round(2*pi / theta1_quantum)
vel_ownship = (100, 1200)
#vel_intruder = (0, 1200) # full range
vel_intruder = (0, 400)

qvimin = round(vel_intruder[0]/vel_quantum)
qvimax = round(vel_intruder[1]/vel_quantum)

qvomin = round(vel_ownship[0]/vel_quantum)
qvomax = round(vel_ownship[1]/vel_quantum)

for i in range(2):
    assert vel_ownship[i] % vel_quantum < 1e-6
    assert vel_intruder[i] % vel_quantum < 1e-6
