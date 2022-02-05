from matplotlib import pyplot as plt
import numpy as np


def showMap(t_list, x_list, y_list, z_list, title='Original Map', ylim=None):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharey=True)
    if (ylim != None):
        plt.ylim(ylim)
    ax[0].set_title(title, fontsize=20)
    colors = ['red', 'green', 'blue']
    lists = [x_list, y_list, z_list]
    plt.figure()
    for i in range(3):
        ax[i].plot(t_list, lists[i], color=colors[i])
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Amplitude')
    fig.show()
