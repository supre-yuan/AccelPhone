import FileRead
import ShowMap
import PreProcessing as pre
import SegmentationPro as seg
import ShowCuttingPro
import pandas as pd
import Spectrogram as sp
import numpy as np


def main():
    prefix = "C:/Users/zhang/Desktop/Phone/XiaoMi/"
    name = "ty_xiaomi_word"
    process(prefix, name)


def process(prefix, name):

    tList, xList, yList, zList = FileRead.fileread_tsv(prefix + name)
    ShowMap.showMap(tList, xList, yList, zList)
    zList = [i * 1 for i in zList]

    t, x, y, z = (tList.copy(), xList.copy(), yList.copy(), zList.copy())

    tList, fList, xList, yList, zList = pre.preprocess(tList, xList, yList, zList, Normalized=1, highpass=1 / 500)
    ShowMap.showMap(tList, xList, yList, zList, 'None Filter')
    tList = np.array(tList)
    xList = np.array(xList)
    yList = np.array(yList)
    zList = np.array(zList)

    zSmooth, tSmooth = seg.smooth(zList, tList)

    cuttingpoints = seg.findCuttingPoints(tSmooth, zSmooth, (0.8, 0.2))

    t, f, x, y, z = pre.preprocess(t, x, y, z, Normalized=1, highpass=1 / 500)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(len(t))

    left = 1000
    right = 1000
    path = "C:/Users/zhang/Desktop/Phone/S9/Picture/" + name + ".png"
    print(path)
    ShowCuttingPro.showcutting(cuttingpoints, t, x, y, z, left, right, path)
    print(len(t))

    print(cuttingpoints)

    length = len(cuttingpoints)
    cnt = 0

    try:
        for i in range(0, 10):
            i = 2 * i
            t_new = t[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            x_new = x[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            y_new = y[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            z_new = z[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            dic = {"z": z_new}
            cnt = cnt + 1
            df = pd.DataFrame(data=dic)
            df.to_csv(prefix + name + "_align_" + str(int(i / 2)) + ".csv", index=False, header=False, sep="\t")
            x_new, y_new, z_new = sp.generateMap(
                prefix + "Spectrogram/" + name + "_Spectrogram_" + str(int(i / 2)) + ".png", x_new, y_new, z_new)
            sp.generateRGB(x_new, y_new, z_new, prefix + "RGB/" + name + "_RGB_" + str(int(i / 2)) + ".png")
    except:
        pass


if __name__ == "__main__":
    main()
