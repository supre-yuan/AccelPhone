import matplotlib.pyplot as plt
import FileRead
import math
import numpy as np
from scipy import signal
import cv2


def generateMap(filename, xList, yList, zList, name="Magnitude Scalogram"):
    try:

        NFFT = 128
        noverlap = 120
        Fs = 500
        fig = plt.figure(figsize=(9, 4))
        plot_Z = fig.subplots(ncols=1)

        fX, tX, specX = signal.stft(xList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)
        fY, tY, specY = signal.stft(yList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)
        fZ, tZ, specZ = signal.stft(zList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)

        specX = np.abs(specX)
        specY = np.abs(specY)
        specZ = np.abs(specZ)

        plot_Z.pcolormesh(tZ, fZ, specZ)
 
        plot_Z.set_xlabel("Time (secs)", fontsize=30)
        plot_Z.set_ylabel("Frequency (Hz)", fontsize=30)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(filename + ".png")

        plt.savefig(filename + ".pdf", bbox_inches='tight')

        return specX, specY, specZ
    except:
        pass


def generateRGB(specX, specY, specZ, name):
    R_x = np.zeros(shape=specX.shape)
    G_y = np.zeros(shape=specX.shape)
    B_z = np.zeros(shape=specX.shape)

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            R_x[i, j] = math.sqrt(specX[i, j])
    maxi = np.max(R_x)
    mini = np.min(R_x)
    R_x = np.ceil((255 / (maxi - mini)) * (R_x - mini))

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            G_y[i, j] = math.sqrt(specY[i, j])
    maxi = np.max(G_y)
    mini = np.min(G_y)
    G_y = np.ceil((255 / (maxi - mini)) * (G_y - mini))

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            B_z[i, j] = math.fabs(specZ[i, j])
    maxi = np.max(B_z)
    mini = np.min(B_z)
    B_z = np.ceil((255 / (maxi - mini)) * (B_z - mini))
    print(np.mean(B_z))

    img = np.zeros(shape=(specX.shape[0], specX.shape[1], 3))
    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            img[i, j, 2] = 0
            img[i, j, 1] = 0
            img[i, j, 0] = B_z[i, j]

    cv2.imwrite(name, img)
