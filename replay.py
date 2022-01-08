'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used to replay traces, as a sanity check that things are working correctly
'''

from functools import lru_cache
import os
import math

import numpy as np
from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

import onnxruntime as ort

from settings import Quanta

skip_quantization = False

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = os.path.join('resources', 'bak_matplotlib.mlpstyle')
    plt.style.use(['bmh', p])

def load_network(last_cmd):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    #print(f"Loading {mat_filename}...")
    #matfile = loadmat(mat_filename)
    #range_for_scaling = matfile['range_for_scaling'][0]
    #means_for_scaling = matfile['means_for_scaling'][0]
    #mat_filename = f"ACASXU_run2a_1_1_batch_2000.mat"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    path = os.path.join("resources", onnx_filename)
    session = ort.InferenceSession(path)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets

def get_time_elapse_mat(command1, dt, command2=0):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy, x2, y2, vx2, vy2 
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    y2 = y_list[command2]
    
    dtheta1 = (y1 / 180 * np.pi)
    dtheta2 = (y2 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 0, 1, 0], # x' = vx
        [0, 0, 0, 0, 0, 0, 0, 1], # y' = vy
        [0, 0, 0, 0, 0, 0, 0, -dtheta2], # vx' = -vy * dtheta2
        [0, 0, 0, 0, 0, 0, dtheta2, 0], # vy' = vx * dtheta1
        ], dtype=float)

    return expm(a_mat * dt)

def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    session, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})
        
    return outputs[0][0]

def quantize(x, delta=50):
    """round to the nearest delta (offset by delta / 2)

    for example using 50 will round anything between 0 and 50 to 25
    """

    global skip_quantization

    if skip_quantization:
        rv = x
    else:
        rv = delta/2 + delta * round((x - delta/2) / delta)

    return rv

def state8_to_qinput_qstate(state8, stdout=False):
    """compute rho, theta, psi from state7"""

    assert len(state8) == 8

    x1, y1, vxo, vyo, x2, y2, vxi, vyi = state8

    pos_quantum = Quanta.pos
    vel_quantum = Quanta.vel
    theta1_quantum = Quanta.theta1

    dy = quantize(y2 - y1, pos_quantum)
    dx = quantize(x2 - x1, pos_quantum)

    rho = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # psi should be 
    theta1 = np.arctan2(vyo, vxo) ## theta 1 is using ownship!!

    if stdout:
        print(f"in state8_to_qstate5, real theta1: {theta1}")

    assert abs(vyi) < 1e-6
    assert vxi >= 0
    theta2 = 0
    theta1 = quantize(theta1, theta1_quantum)

    if stdout:
        print(f"quantized theta1: {theta1}")

    theta1_deg = theta1 * 360/(2*math.pi)

    if stdout:
        print(f"theta1 quantized: {round(theta1_deg, 3)} deg")
        print(f"ownship vx / vy = {vxo}, {vyo}")

    psi = theta2 - theta1

    theta -= theta1

    while theta < -np.pi:
        theta += 2 * np.pi

    while theta > np.pi:
        theta -= 2 * np.pi

    if psi < -np.pi:
        psi += 2 * np.pi

    while psi > np.pi:
        psi -= 2 * np.pi

    own_vel = math.sqrt(vxo**2 + vyo**2)
    int_vel = math.sqrt(vxi**2)

    q_v_own = quantize(own_vel, vel_quantum)
    q_v_int = quantize(int_vel, vel_quantum)

    if stdout:
        print(f"dx: {dx}")
        print(f"dy: {dy}")
        print(f"theta1: {theta1}")
        print(f"theta2: {theta2}")
        print(f"q_v_own: {q_v_own} (from {own_vel})")
        print(f"q_v_int: {q_v_int}")

    qinput = np.array([rho, theta, psi, q_v_own, q_v_int])
    qstate = (dx, dy, theta1, theta2, q_v_own, q_v_int)

    return qinput, qstate

def state7_to_state8(state7, v_own, v_int):
    """compute x,y, vx, vy, x2, y2, vx2, vy2 from state7"""

    assert len(state7) == 7

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v_own
    vy1 = math.sin(state7[2]) * v_own

    x2 = state7[3]
    y2 = state7[4]
    vx2 = math.cos(state7[5]) * v_int
    vy2 = math.sin(state7[5]) * v_int

    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""
    
    img = plt.imread(os.path.join('resources', 'airplane.png'))

    return img

def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []

    for cmd in range(5):
        rv.append([])
        
        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            rv[-1].append(mat)

    return rv

class State:
    'state of execution container'

    nets = load_networks()
    plane_size = 2500

    nn_update_rate = 1.0
    dt = 1.0

    time_elapse_mats = init_time_elapse_mats(dt)

    def __init__(self, init_vec, save_states=False):
        assert len(init_vec) == 8, "init vec should have length 8"
        
        self.state8 = np.array(init_vec, dtype=float) # current state
        self.next_nn_update = 0.0
        self.command = 0 # initial command

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.vec_list = [] # state history
        self.commands = [] # commands history
        self.int_commands = [] # intruder command history

        self.qinputs = []

        # used only if plotting
        self.artists_dict = {} # set when make_artists is called
        self.img = None # assigned if plotting

        # assigned by simulate()
        self.u_list = []
        self.u_list_index = None
        self.min_dist = np.inf

    def artists_list(self):
        'return list of artists'

        return list(self.artists_dict.values())

    def set_plane_visible(self, vis):
        'set ownship plane visibility status'

        self.artists_dict['dot0'].set_visible(not vis)
        self.artists_dict['circle0'].set_visible(False) # circle always False
        self.artists_dict['lc0'].set_visible(True)
        self.artists_dict['plane0'].set_visible(vis)
        
    def update_artists(self, axes):
        '''update artists in self.artists_dict to be consistant with self.vec, returns a list of artists'''

        assert self.artists_dict
        rv = []

        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = self.state8
        theta1 = math.atan2(vy1, vx1)
        theta2 = math.atan2(vy2, vx2)

        for i, x, y, theta in zip([0, 1], [x1, x2], [y1, y2], [theta1, theta2]):
            key = f'plane{i}'

            if key in self.artists_dict:
                plane = self.artists_dict[key]
                rv.append(plane)

                if plane.get_visible():
                    theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                    original_size = list(self.img.shape)
                    img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                    rotated_size = list(img_rotated.shape)
                    ratios = [r / o for r, o in zip(rotated_size, original_size)]
                    plane.set_data(img_rotated)

                    size = State.plane_size
                    width = size * ratios[0]
                    height = size * ratios[1]
                    box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                    tbox = TransformedBbox(box, axes.transData)
                    plane.bbox = tbox

            key = f'dot{i}'
            if key in self.artists_dict:
                dot = self.artists_dict[f'dot{i}']
                cir = self.artists_dict[f'circle{i}']
                rv += [dot, cir]

                dot.set_data([x], [y])
                cir.set_center((x, y))

        # line collection
        lc = self.artists_dict['lc0']
        rv.append(lc)

        int_lc = self.artists_dict['int_lc0']
        rv.append(int_lc)

        self.update_lc_artists(lc, int_lc)

        return rv

    def update_lc_artists(self, own_lc, int_lc):
        'update line collection artist based on current state'

        assert self.vec_list

        for lc_index, lc in enumerate([own_lc, int_lc]):
            paths = lc.get_paths()
            colors = []
            lws = []
            paths.clear()
            last_command = -1
            codes = []
            verts = []

            for i, vec in enumerate(self.vec_list):
                if np.linalg.norm(vec - self.state8) < 1e-6:
                    # done
                    break

                if lc_index == 0:
                    cmd = self.commands[i]
                else:
                    cmd = self.int_commands[i]

                x = 0 if lc_index == 0 else 4
                y = 1 if lc_index == 0 else 5

                # command[i] is the line from i to (i+1)
                if cmd != last_command:
                    if codes:
                        paths.append(Path(verts, codes))

                    codes = [Path.MOVETO]
                    verts = [(vec[x], vec[y])]

                    if cmd == 1: # weak left
                        lws.append(2)
                        colors.append('b')
                    elif cmd == 2: # weak right
                        lws.append(2)
                        colors.append('c')
                    elif cmd == 3: # strong left
                        lws.append(2)
                        colors.append('g')
                    elif cmd == 4: # strong right
                        lws.append(2)
                        colors.append('r')
                    else:
                        assert cmd == 0 # coc
                        lws.append(2)
                        colors.append('k')

                codes.append(Path.LINETO)

                verts.append((self.vec_list[i+1][x], self.vec_list[i+1][y]))

            # add last one
            if codes:
                paths.append(Path(verts, codes))

            lc.set_lw(lws)
            lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list
        self.img = get_airplane_img()

        posa_list = [(v[0], v[1], math.atan2(v[3], v[2])) for v in self.vec_list]
        posb_list = [(v[4], v[5], math.atan2(v[7], v[6])) for v in self.vec_list]
        
        pos_lists = [posa_list, posb_list]

        if show_intruder:
            pos_lists.append(posb_list)

        for i, pos_list in enumerate(pos_lists):
            x, y, theta = pos_list[0]
            
            l = axes.plot(*zip(*pos_list), f'c-', lw=0, zorder=1)[0]
            l.set_visible(False)
            self.artists_dict[f'line{i}'] = l

            if i == 0:
                lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(lc)
                self.artists_dict[f'lc{i}'] = lc

                int_lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(int_lc)
                self.artists_dict[f'int_lc{i}'] = int_lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], 'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 1500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c

    def step(self, stdout=False):
        'execute one time step and update the model'

        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                      f"next update: {self.next_nn_update}"

            if stdout:
                print(f"\nupdating nn command. Using network #{self.command}")

            # update command
            self.update_command(stdout=stdout)

            if stdout:
                print(f"nn output cmd was {self.command}")

            self.next_nn_update = State.nn_update_rate

        self.next_nn_update -= State.dt
        intruder_cmd = self.u_list[self.u_list_index]

        if self.save_states:
            self.commands.append(self.command)
            self.int_commands.append(intruder_cmd)

        time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd]

        self.state8 = time_elapse_mat @ self.state8

        if stdout:
            print(f"state8: {self.state8}")

    def simulate(self, cmd_list, stdout=False):
        '''simulate system

        saves result in self.vec_list
        also saves self.min_dist
        '''

        self.u_list = cmd_list
        self.u_list_index = None

        assert isinstance(cmd_list, list)
        tmax = len(cmd_list) * State.nn_update_rate

        t = 0.0

        if self.save_states:
            rv = [self.state8.copy()]

        #self.min_dist = 0, math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2), self.vec.copy()
        prev_dist_sq = (self.state8[0] - self.state8[4])**2 + (self.state8[1] - self.state8[5])**2

        while t + 1e-6 < tmax:
            self.step(stdout=stdout)

            cur_dist_sq = (self.state8[0] - self.state8[4])**2 + (self.state8[1] - self.state8[5])**2

            if self.save_states:
                rv.append(self.state8.copy())

            t += State.dt

            prev_dist_sq = cur_dist_sq
            
        self.min_dist = math.sqrt(prev_dist_sq)
        
        if self.save_states:
            self.vec_list = rv

        if not self.save_states:
            assert not self.vec_list
            assert not self.commands
            assert not self.int_commands

    def update_command(self, stdout=False):
        'update command based on current state'''

        qinput, qstate = state8_to_qinput_qstate(self.state8, stdout=stdout)
        rho, theta, psi, v_own, v_int = qinput

        if stdout:
            print(f"state8: {self.state8}")
            print(f"qinput: {rho, theta, psi, v_own, v_int}")

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200
        last_command = self.command

        if rho > 60760:
            self.command = 0
        else:
            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state)
            self.command = np.argmin(res)

            #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        if self.u_list_index is None:
            self.u_list_index = 0
        else:
            self.u_list_index += 1

            # repeat last command if no more commands
            self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)

        self.qinputs.append((last_command, self.state8, qstate, qinput, self.command))

def plot(s, save_mp4=False):
    """plot a specific simulation"""
    
    init_plot()
    dx = s.state8[0] - s.state8[4]
    dy = s.state8[1] - s.state8[5]
    dist = math.sqrt(dx*dx + dy*dy)
    print(f"Plotting state with min_dist: {round(s.min_dist, 2)} and " + \
          f"final dx: {round(dx, 1)}, dy: {round(dy, 1)}, dist: {round(dist, 2)}")
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes.axis('equal')

    axes.set_title("ACAS Xu Simulation")
    axes.set_xlabel('X Position (ft)')
    axes.set_ylabel('Y Position (ft)')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=14,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(True)

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
                fontsize=14, loc='lower left')
    
    s.make_artists(axes, show_intruder=True)
    states = [s]

    plt.tight_layout()

    num_steps = len(states[0].vec_list)
    interval = 50 # ms per frame
    freeze_frames = 2 if not save_mp4 else 5

    num_runs = 1 # 3
    num_frames = num_runs * num_steps + 2 * num_runs * freeze_frames

    print(f"num_frames: {num_frames}")

    #plt.savefig('plot.png')
    #plot_commands(states[0])

    def animate(f):
        'animate function'

        if (f+1) % 10 == 0 and save_mp4:
            print(f"Frame: {f+1} / {num_frames}")

        run_index = f // (num_steps + 2 * freeze_frames)

        f = f - run_index * (num_steps + 2*freeze_frames)

        f -= freeze_frames

        f = max(0, f)
        f = min(f, num_steps - 1)

        num_states = len(states)

        if f == 0:
            # initiaze current run_index
            show_plane = num_states <= 10
            for s in states[:num_states]:
                s.set_plane_visible(show_plane)

            for s in states[num_states:]:
                for a in s.artists_list():
                    a.set_visible(False)

        time_text.set_text(f'Time: {f * State.dt:.1f}')

        artists = [time_text]

        for s in states[:num_states]:
            s.state8 = s.vec_list[f]
            artists += s.update_artists(axes)

        for s in states[num_states:]:
            artists += s.artists_list()

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('sim.mp4', writer=writer)
    else:
        plt.show()

def main():
    'main entry point'

    global skip_quantization
    try_without_quantization = True
    
    ###################
    alpha_prev_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    qtheta1 = 311
    qv_own = 11
    qv_int = 39
    # chebeshev center radius: 0.0745862895428716
    end = np.array([ -99.92541371, -499.92541371,   74.1164437 ,  -82.42611738,
              0.        ,  390.10329256])
    start = np.array([ -5360.83116819,   7007.87669426,    -65.21523383,    -89.63417501,
           -40570.74242582,    390.10329256])

    ##################

    skip_checks = True
        
    if try_without_quantization:
        skip_quantization = True
        skip_checks = True
        #alpha_prev_list = []

    theta1_quantum = Quanta.theta1
        
    q_theta1 = qtheta1 * theta1_quantum + theta1_quantum / 2 
    cmd_list = [0] * (len(alpha_prev_list) - 1)

    _, _, vx, vy, _, vxi = start

    own_vel = math.sqrt(vx**2 + vy**2)
    int_vel = math.sqrt(vxi**2)

    print(f"init own vel: {own_vel}")
    print(f"init int vel: {int_vel}")

    if not skip_quantization and not skip_checks:
        # double-check quantization matches expectation
        
        print(f"own_vel computed: {own_vel}, quantized: {qv_own}")
        print(f"int_vel computed: {int_vel}, quantized: {qv_int}")

        print(f"ownship vx / vy = {vx}, {vy}")

        theta1 = math.atan2(vy, vx)
        print(f"real theta1: {theta1}")
        theta1_deg = theta1 * 360/(2*math.pi)
        q_theta1_deg = q_theta1 * 360/(2*math.pi)
        print(f"q_theta1 computed: {round(theta1_deg, 3)} deg, quantized: {round(q_theta1_deg, 3)} deg")

        actual_qtheta1 = quantize(theta1, theta1_quantum)
        if actual_qtheta1 < 0:
            actual_qtheta1 += 2 * math.pi
            
        print(f"actual_qtheta1: {actual_qtheta1}")
        actual_qtheta1_deg = actual_qtheta1 * 360/(2*math.pi)

        print(f"actual_qtheta1_deg = {actual_qtheta1_deg}")
        assert abs(actual_qtheta1 - q_theta1) < 1e-4, f"qtheta1 was actually {round(actual_qtheta1_deg, 3)}, " + \
            f"expected {round(q_theta1_deg, 3)}"

    init_vec = [start[0], start[1], start[2], start[3], start[4], 0, start[5], 0]

    # run time backwards N seconds
    rewind_seconds = 40

    if rewind_seconds != 0:
        a_mat = get_time_elapse_mat(0, -rewind_seconds)
        init_vec = a_mat @ init_vec

        cmd_list = [cmd_list[0]] * rewind_seconds + cmd_list
    ########
    
    # run the simulation
    s = State(init_vec, save_states=True)

    s.command = alpha_prev_list[-1]
    
    s.simulate(cmd_list, stdout=False)
    print("Simulation completed.\n")

    if not skip_checks:

        # extra printing on trace
        if False:
            for i, (net, state8, qstate, qinput, cmd_out) in enumerate(s.qinputs):
                print(f"{i+1}. network {net} with qinput: {tuple(q for q in qinput)} -> {cmd_out}")
                print(f"state: {tuple(x for x in state8)}")
                print(f"qstate: {[x for x in qstate]}")

        expected_end = np.array([end[0], end[1], end[2], end[3], end[4], 0, end[5], 0])
        print(f"expected end: {expected_end}")
        print(f"actual end: {s.state8}")

        print(f"cmds: {s.commands}, alpha_prev_list: {list(reversed(alpha_prev_list[:-1]))}")

        assert s.commands == list(reversed(alpha_prev_list[:-1])), "command mismatch"
        print("commands matched!")

        difference = np.linalg.norm(s.state8 - expected_end, ord=np.inf)
        print(f"end state difference: {difference}")
        assert difference < 1e-2, f"end state mismatch. difference was {difference}"
        print("end states were close enough")
    else:
        got_cmds = s.commands[:3]
        expected_cmds = list(reversed(alpha_prev_list[:-1]))[:3]
        
        print(f"Got first few commands: {got_cmds}")
        print(f"Expected first few commands: {expected_cmds}")
        
        print("WARNING: skipped sanity checks on replay")

    if rewind_seconds != 0:
        print(f"commands: {s.commands}")
        print("WARNING: rewind_seconds != 0")
        
    # optional: do plot
    #plot(s, save_mp4=False)
    plot_image(s)

if __name__ == "__main__":
    main()
