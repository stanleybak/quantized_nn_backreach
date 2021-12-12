"""
plotting for quantized backreach
"""

from functools import lru_cache

import numpy as np

import matplotlib.pyplot as plt
from star import Star

class Plotter:
    """object in charge of plotting"""

    def __init__(self):

        self.fig, self.ax_list = plt.subplots(2, 3, figsize=(12, 8))

        for ax_row in self.ax_list:
            for ax in ax_row:
                ax.axis('equal')

        p = 'resources/bak_matplotlib.mlpstyle'
        plt.style.use(['bmh', p])

    def plot_quantization(self, qlist):
        """plot quantized states"""

        dxs = []
        dys = []
        vxos = []
        vyos = []
        #vxis = []

        for qstate, qstar in qlist:
            dx, dy, vxo, vyo, _vxi, _, _ = qstate

            dxs.append(dx)
            dys.append(dy)
            vxos.append(vxo)
            vyos.append(vyo)
            #vxis.append(vxi)

            # plot qstar
            self.plot_star(qstar, 'r')

        # dx/dy plot
        ax = self.ax_list[1][0]
        ax.plot(dxs, dys, "go")

        # vxo/vyo plot
        ax = self.ax_list[0][1]
        ax.plot(vxos, vyos, "go")

    def plot_star(self, star, color='k'):
        """add the current state to the plot"""

        labels = ("X_own", "Y_own", "VX_own", "VY_own", "X_int", "VX_int")
        index = 0

        for index in range(3):
            ax = self.ax_list[0][index]

            ax.set_xlabel(labels[2*index])
            ax.set_ylabel(labels[2*index + 1])

            verts = star.verts(2*index, 2*index + 1)
            ax.plot(*zip(*verts), '-', color=color, zorder=1)

        # plot 4: deltax / deltay
        # dx = x_int - x_own
        ax = self.ax_list[1][0]

        ax.set_xlabel("dx (x_int - x_own)")
        ax.set_ylabel("dy (y_int - y_own)")
            
        xdim = np.zeros(Star.NUM_VARS)
        xdim[Star.X_INT] = 1
        xdim[Star.X_OWN] = -1

        ydim = np.zeros(Star.NUM_VARS)
        #ydim[Star.Y_INT] = 1 # Y_int is always 0
        ydim[Star.Y_OWN] = -1

        verts = star.verts(xdim, ydim)
        ax.plot(*zip(*verts), '-', color=color, zorder=1)

        # plot 5: deltavx / deltavy
        # dvx = vx_int - vx_own
        ax = self.ax_list[1][1]

        ax.set_xlabel("dvx (vx_int - vx_own)")
        ax.set_ylabel("dvy (vy_int - vy_own)")
            
        xdim = np.zeros(Star.NUM_VARS)
        xdim[Star.VX_INT] = 1
        xdim[Star.VX_OWN] = -1

        ydim = np.zeros(Star.NUM_VARS)
        #ydim[Star.Y_INT] = 1 # Y_int is always 0
        ydim[Star.VY_OWN] = -1

        verts = star.verts(xdim, ydim)
        ax.plot(*zip(*verts), '-', color=color, zorder=1)

        # plot 6: x and y position of both int and ownship
        ax = self.ax_list[1][2]

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        verts = star.verts(Star.X_OWN, Star.Y_OWN)
        ax.plot(*zip(*verts), 'k-', zorder=1)

        ydim = np.zeros(Star.NUM_VARS)
        verts = star.verts(Star.X_INT, ydim)
        ax.plot(*zip(*verts), 'b-o', zorder=1)

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""
    
    img = plt.imread('airplane.png')

    return img

