"""
plotting for quantized backreach
"""

from functools import lru_cache

import matplotlib.pyplot as plt

class Plotter:
    """object in charge of plotting"""

    def __init__(self):

        self.fig, self.ax_list = plt.subplots(2, 2, figsize=(8, 8))

        p = 'resources/bak_matplotlib.mlpstyle'
        plt.style.use(['bmh', p])

    def plot_state(self, s, color='k-'):
        """add the current state to the plot"""

        labels = ("X_own", "Y_own", "VX_own", "VY_own", "X_int", "VX_int")
        index = 0

        for y in [0, 1]:
            for x in [0, 1]:
                if x == 1 and y == 1:
                    break

                ax = self.ax_list[y][x]

                ax.set_xlabel(labels[index])
                ax.set_ylabel(labels[index + 1])

                verts = s.star.verts(index, index + 1)
                ax.plot(*zip(*verts), color, zorder=1)
                
                index += 2


@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""
    
    img = plt.imread('airplane.png')

    return img

