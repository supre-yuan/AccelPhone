import DataRemove
import Standardize
import Interp
import HighPassFilter


def preprocess(tList, xList, yList, zList, Normalized=0, highpass=-1):
    tList, xList, yList, zList = DataRemove.dataRemove(tList, xList, yList, zList, start=4000, end=-1000)

    if Normalized == 1:
        tList, xList, yList, zList = Standardize.standardize(tList, xList, yList, zList)

    tList, xList, yList, zList = Interp.interp(tList, xList, yList, zList)

    if highpass >= 0:
        tList, fList, xList, yList, zList = HighPassFilter.highPassFilter(xList, yList, zList, highpass)
        return tList, fList, xList, yList, zList
    return tList, xList, yList, zList
