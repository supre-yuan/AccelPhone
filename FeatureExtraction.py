from asyncio import sleep
import FeatureExtraction_PreProcessing as pre
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def getMean(z):
    result = np.mean(z)
    return result


def getAbsMean(z):
    result = np.mean(np.abs(z))
    return result


def getStdDev(z):
    result = np.std(z)
    return result


def getAbsAveDev(z):
    result = np.sum(np.abs(z - np.mean(z))) / len(z)
    return result


def getSkewness(z):
    z = pd.Series(z)
    return z.skew()


def getKurtosis(z):
    z = pd.Series(z)
    return z.kurtosis()


def getRMS(z):
    result = np.sqrt(np.mean(z ** 2))
    return result


def getQ1(z):
    result = z[math.ceil(1 * len(z) / 4)]
    return result


def getQ1pro(z):
    result = 0
    for i in range(-50, 51):
        result = result + z[math.ceil(1 * len(z) / 4) + i]
    result = result / 100
    return result


def getQ2(z):
    result = z[math.ceil(2 * len(z) / 4)]
    return result


def getQ2pro(z):
    result = 0
    for i in range(-50, 51):
        result = result + z[math.ceil(2 * len(z) / 4) + i]
    result = result / 100
    return result


def getQ3(z):
    result = z[math.ceil(3 * len(z) / 4)]
    return result


def getQ3pro(z):
    result = 0
    for i in range(-50, 51):
        result = result + z[math.ceil(3 * len(z) / 4) + i]
    result = result / 100
    return result


def getTotalSVM(x, y, z):
    return getRMS(x + y + z)


def getSpecCentroid(fz, freqList):
    zm = fz
    zf = freqList
    C = np.sum(zf * zm) / np.sum(zm)
    return C


def getSpecStdDev(fz, freqList):
    zm = fz
    zf = freqList
    C = getSpecMean(fz, freqList)
    d = np.sqrt(np.sum(((zf - C) ** 2) * zm) / np.sum(zm))
    return d


def getSpecSkewness(fz, freqList):
    zm = fz
    zf = freqList
    C = getSpecMean(fz, freqList)
    d = getSpecStdDev(fz, freqList)
    gamma = (np.sum((zf - C) ** 3 * zm)) / d ** 3
    return gamma


def getSpecKurt(fz, freqList):
    zm = fz
    zf = freqList
    C = getSpecMean(fz, freqList)
    d = getSpecStdDev(fz, freqList)
    beta = (np.sum((zf - C) ** 4 * zm)) / d ** 4 - 3
    return beta


def getSpecCrest(fz, freqList):
    zm = fz
    zf = freqList
    C = getSpecMean(fz, freqList)
    CR = np.max(zm) / C
    return CR


def getSpectrumPeak(fz, freqlist):
    max_magnitude = 0
    max_freq = 0
    for i in range(len(fz)):
        if (max_magnitude < fz[i]):
            max_magnitude = fz[i]
            max_freq = freqlist[i]
    return max_magnitude, max_freq


def getHps(fz, freqList):
    data = fz
    n = len(fz) / 2
    M = np.ceil(n / 3)
    if (M <= 1):
        return 0
    peak_index = 0
    peak = 0
    for i in range(0, int(M)):
        tempProduct = data[i] * data[i * 2] * data[i * 3]
        if (tempProduct > peak):
            peak = tempProduct
            peak_index = i
    largest1_lwr = position1_lwr = 0
    for i in range(0, len(fz)):
        if (data[i] > largest1_lwr and i != peak_index):
            largest1_lwr = data[i]
            position1_lwr = i
    ratio1 = data[position1_lwr] / data[peak_index]
    if (position1_lwr > peak_index * 0.4 and position1_lwr < peak_index * 0.6 and ratio1 > 0.1):
        peak_index = position1_lwr
    result = data[int(n + peak_index)]
    return result


def getSharpness(fz, freqList):
    n = len(fz)
    temp = 0
    for i in range(0, n):
        sl = fz[i] ** 0.23
        g = 1 if i < 15 else 0.066 * np.exp(0.171 * i)
        temp = temp + n * g * sl
    result = temp
    return result


def getCrest(z):
    maxmax = np.max(z)
    meanmean = np.mean(z)
    if meanmean != 0:
        result = maxmax / meanmean
        return result
    else:
        return 0


def getSpecMean(fz, freqList):
    zm = fz
    zf = freqList
    C = np.sum(zf * zm) / np.sum(zm)
    return C


def getAveDev(z):
    result = np.sum(z - np.mean(z)) / len(z)
    return result


def getVar(z):
    result = np.var(z)
    return result


def getSpecVar(fz, freqList):
    zm = fz
    zf = freqList
    C = getSpecMean(fz, freqList)
    V = np.sum(((zf - C) ** 2) * zm) / np.sum(zm)
    return V


def getIrregularityK(z):
    N = len(z)
    M = N - 1
    result = 0
    for i in range(1, M):
        result = result + np.abs(z[i] - (z[i - 1] + z[i] + z[i + 1]) / 3)
    return result


def getIrregularityJ(z):
    N = len(z)
    n = N - 1
    num = 0
    den = 0
    for i in range(0, n):
        num = num + ((z[i] - z[i + 1]) ** 2)
        den = den + z[i] ** 2
    result = num / den
    return result


def getSpread(fz, freqList):
    result = getSpecVar(fz, freqList)
    return result


def getZCR(z):
    n = len(z)
    result = 0
    for i in range(1, n):
        if (z[i] * z[i - 1] < 0):
            result = result + 1
    result = result / n
    return result


def getNoisiness(fz, freqList):
    zm = fz
    h = np.max(zm)
    p = getSpecCentroid(fz, freqList)
    i = p - h
    result = i / p
    return result


def getSpecSlope(fz, freqList):
    zm = fz
    zf = freqList
    n = M = len(zm) / 2
    F = A = FA = FXTRACT_SQ = 0
    for i in range(0, int(n)):
        f = zf[i]
        a = zm[i]
        F = F + f
        A = A + a
        FA = FA + f * a
        FXTRACT_SQ = FXTRACT_SQ + f * f
    result = (1 / A) * (M * FA - F * A) / (M * FXTRACT_SQ - F * F)
    return result


def getLowestValue(z):
    result = np.min(z)
    return result


def getHighestValue(z):
    result = np.max(z)
    return result


def getSum(z):
    result = np.sum(z)
    return result


def getNonzeroCount(z):
    result = 0
    for i in range(0, len(z)):
        if z[i] != 0:
            result = result + 1
    return result


def getSmoothness(z):
    M = len(z) - 1
    current = next = temp = 0
    XTRACT_LOG_LIMIT = 2e-42
    for i in range(1, M):
        if i == 1:
            prev = XTRACT_LOG_LIMIT if z[i - 1] <= 0 else z[i - 1]
            current = XTRACT_LOG_LIMIT if z[i] <= 0 else z[i]
        else:
            prev = current
            current = next
        next = XTRACT_LOG_LIMIT if z[i + 1] <= 0 else z[i + 1]
        temp = temp + np.abs(20 * np.log(current) - (20 * np.log(prev) + 20 * np.log(current) + 20 * np.log(next)) / 3)
    result = temp
    return result


def extract():
    prefix = r"C:\Users\Spure_yuan\Desktop\SupreYuanCode\Data_0929\S8"
    postfix = r"ty_s8_word_align"
    name = postfix.split("_")
    word = name[2]

    for i in range(10):
        STR = prefix + "\\" + postfix + "_" + str(i) + ".csv"

        tList, xList, yList, zList = pre.readFile(STR)
        print("xList的值为：", xList)
        print("yList的值为：", yList)
        print("zList的值为：", zList)

        freqList, xList, yList, zList = pre.highPassFilter(xList, yList, zList, thresRate=30 / 500)

        xList, yList, zList = pre.reverseFFT(xList, yList, zList)
        # TODO: =======================TIME DOMAIN=======================
        t = []
        t.append(getMean(zList))  # 1
        t.append(getStdDev(zList))  # 2
        t.append(getKurtosis(zList))  # 3
        t.append(getSkewness(zList))  # 4
        t.append(getAbsAveDev(zList))  # 5
        t.append(getRMS(zList))  # 6
        t.append(getQ1(zList))  # 7
        t.append(getQ2(zList))  # 8
        t.append(getQ3(zList))  # 9
        t.append(getTotalSVM(xList, yList, zList))  # 10
        t.append(getAbsMean(zList))  # 11
        t.append(getVar(zList))  # 12
        t.append(getIrregularityK(zList))  # 13
        t.append(getIrregularityJ(zList))  # 14
        t.append(getZCR(zList))  # 15
        t.append(getLowestValue(zList))  # 16
        t.append(getHighestValue(zList))  # 17
        t.append(getSum(zList))  # 18
        t.append(getNonzeroCount(zList))  # 19
        t.append(getSmoothness(zList))  # 20
        # t.append(getLoudness(zList))  #
        t.append(getAveDev(zList))  # 21
        t.append(getCrest(zList))  # 22
        # t.append(getAutocorreletion(zList))  #
        t.append(getQ1pro(zList))  # 23
        t.append(getQ2pro(zList))  # 24
        # t.append(getQ3pro(zList))  # 25
        t = np.array(t)
        # TODO: =======================FREQ DOMAIN=======================
        f = []
        length = len(zList)
        zList = zList[0:round(length / 2)]
        freqList = freqList[0:round(length / 2)]
        f.append(getSpecStdDev(np.abs(zList), freqList))  # 26
        f.append(getSpecCentroid(np.abs(zList), freqList))  # 27
        f.append(getSpecSkewness(np.abs(zList), freqList))  # 28
        f.append(getSpecKurt(np.abs(zList), freqList))  # 29
        f.append(getSpecCrest(np.abs(zList), freqList))  # 30
        f.append(getSpecVar(np.abs(zList), freqList))  # 31
        f.append(getSpread(np.abs(zList), freqList))  # 32
        f.append(getNoisiness(np.abs(zList), freqList))  # 33
        f.append(getSpecSlope(np.abs(zList), freqList))  # 34
        f.append(getSpecMean(np.abs(zList), freqList))  # 35
        f.append(getSharpness(np.abs(zList), freqList))  # 36
        f.append(getHps(np.abs(zList), freqList))  # 37
        max_magnitude, max_freq = getSpectrumPeak(np.abs(zList), freqList)
        f.append(max_magnitude)  # 38
        f.append(max_freq)  # 39

        if (word == "pswd"):
            f.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if (word == "key"):
            f.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if (word == "secret"):
            f.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if (word == "code"):
            f.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if (word == "account"):
            f.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if (word == "salary"):
            f.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if (word == "encoder"):
            f.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if (word == "bank"):
            f.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if (word == "number"):
            f.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if (word == "word"):
            f.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        f = np.array(f)
        result = np.concatenate((t, f), axis=0)
        DF = pd.DataFrame(data=result).T
        path = r"C:\Users\Spure_yuan\Desktop\SupreYuanCode\Data_0929\S8\Feature"
        savedName = path + '\\' + "S8_feature_Pro.csv"
        DF.to_csv(path_or_buf=savedName, header=False, index=False, mode='a', sep=',')


if __name__ == "__main__":
    extract()