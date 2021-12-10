"""
plotting for quantized backreach
"""

from functools import lru_cache

import matplotlib as plt

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = 'resources/bak_matplotlib.mlpstyle'
    plt.style.use(['bmh', p])

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""
    
    img = plt.imread('airplane.png')

    return img
