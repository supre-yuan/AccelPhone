import numpy as np
import matplotlib.pyplot as plt


def smooth(zList, tList):
    zList_tmp = zList.copy()
    length = len(zList)
    config1 = 300
    config2 = 300
    newSize = config1 + config2 - 2
    tSmooth = tList[(int)(newSize / 2):length - (int)(newSize / 2)]
    zSmooth = [0 for i in range(0, length - newSize)]
    zList_tmp[:] = np.abs(zList_tmp)
    zSmooth[:] = np.convolve(zList_tmp, np.ones(config1) / config1, mode='valid')
    zSmooth[:] = np.convolve(zSmooth, np.ones(config2) / config2, mode='valid')
    return zSmooth, tSmooth


def findCuttingPoints(tSmooth, zSmooth, para):
    zSmooth = np.array(zSmooth)
    Mmax = np.max(zSmooth)
    Mmin = np.min(zSmooth)
    leng = zSmooth.size
    thres = para[0] * Mmin + para[1] * Mmax
    cuttingpoints = []
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 3), sharey=True)
    # plt.xticks(np.linspace(0, 35000, 10))
    ax.set_title('Smoothed signal along the z-axis', fontsize=30)
    plt.ylabel('Acc (m/s\N{SUPERSCRIPT TWO})', fontsize=30, labelpad=34)
    plt.xlabel('Time (secs)', fontsize=30)
    # labelpad=33.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    y = []
    for i in range(0, len(tSmooth)):
        y.append(thres)
    ax.plot(tSmooth, zSmooth, lw=5)
    ax.plot(tSmooth, y, lw=5)

    for i in range(0, leng - 1):
        if zSmooth[i] <= thres and zSmooth[i + 1] > thres:
            cuttingpoints.append(i)
        elif zSmooth[i] >= thres and zSmooth[i + 1] < thres:
            cuttingpoints.append(i)

    # cuttingpoints = [2845, 3777, 6591, 7727, 10859, 12051, 15206, 16302, 19510, 20584, 23802, 24851, 28185, 29193, 32542, 33514, 36814, 37991, 41313, 42263, 45437, 46521, 49758, 50793]
    for i in cuttingpoints:
        plt.scatter(tSmooth[i], zSmooth[i], color='red', linewidths=10)
    # ax.figure.savefig("smooth.pdf", bbox_inches='tight')
    plt.figure(figsize=(16, 3))
    fig.show()
    return cuttingpoints
