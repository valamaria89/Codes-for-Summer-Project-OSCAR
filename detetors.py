import numpy as np
import pandas as pd
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import fnmatch
import os
import re

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def getFiles(pdir):
    fnames = []
    for file in os.listdir(pdir):
        if fnmatch.fnmatch(file, "run*.dat"):
            fnames.append(os.path.join(pdir, file))
    return fnames


def getInt(string):
    try:
        return int(''.join(filter(str.isdigit, string)))
    except ValueError:
        return np.nan

def getfloat(string):
    try:
        res = re.findall(r"\d+[\.\,]\d+", string)[0]
        res = res.replace(',', '.')
        return float(res)
    except (ValueError, IndexError):
        return np.nan


def sortStringAndInt(names):
    # a = int(filter(str.isdigit, names))
    Nmax = max([len(n) for n in names])
    mlist = [(getInt(name), name) for name in names]
    # arr = np.array(mlist)
    arr = np.array(mlist, dtype=[("num", "i"), ("name", "U{}".format(Nmax))])
    # dtype="i, U{}".format(Nmax)
    arr = np.sort(arr, axis=0)
    return arr["name"]


def readDetData(fname):
    """
    Read data of format_
    ---
    Date  19.jun  10.39
    Temp  NAN
    Channel number  Centrod1  Centroid2 Centroid3 Centroid4 Resolution 1332 keV peak
    0 298.337195  876.124441  2953.189232 3348.186431 2,16
    ---
    """
    header = ["det", "c1", "c2", "c3", "c4", "r1332"]
    data = pd.read_csv(fname, header=None, names=header,
                       skiprows=3, sep='\s+')

    # raise detector numbers for Frank:
    data["det"] += 1

    with open(fname) as fp:
        for i, line in enumerate(fp):
            #print(fname)
            if i == 0:
                line = line.split(" ")
                try:
                    #print(line[1:])
                    fmt = " %d.%b  %H:%M\n"
                    time = pd.to_datetime(' '.join(line[1:]),
                                          format=fmt)
                except ValueError:
                    try:
                        fmt = " %d.%b  %H.%M\n"
                        time = pd.to_datetime(' '.join(line[1:]),
                                              format=fmt)
                    except ValueError:
                        raise NotImplementedError("Check date/time format")

            elif i == 1:
                T = getfloat(line)
            else:
                break
    return time, T, data


def map_iterator_to_grid(counter, Nx):
    # Returns i, j coordinate pairs to map a single iterator onto a 2D grid for subplots.
    # Counts along each row from left to right, then increments row number
    i = counter // Nx
    j = counter % Nx
    return i, j


def plotAtts(data, att):
    print("Start print {}".format(att))

    f_mat, ax_mat = plt.subplots(3, 2, figsize=(9, 10), sharex=True,
                                 gridspec_kw={'hspace': 0})

    for i in range(Ndets):
        fig_i, ax_i = plt.subplots()

        # individual plots
        x = range(len(data.dataset))
        y = data[att][:, i]
        ax_i.plot(x, y)

        # combined plots
        ax_comb = ax_mat.flat[int(i/5)]
        y_comb = y-y.mean()
        ax_comb.plot(x, y_comb, "o-", label="det {}".format(i))
        ax_comb.legend(loc="best", prop={'size': 6})

        ax_i.set_xlabel("run")
        ax_i.set_ylabel("channel")
        ax_comb.set_xlabel("run")
        ax_comb.set_ylabel("deviation from mean channel", fontsize=5)

        # delete from memory
        fig_i.suptitle("det {}".format(i))
        fig_i.savefig("figs/run/T_vs_{}_det{:2d}".format(att, i))
        plt.close(fig_i)
    print("Done")
    # plt.tight_layout()
    f_mat.savefig("figs/run/T_vs_{}_combined".format(att))
    plt.close("all")
    plt.show()


def plotAttsTime(data, att):
    print("Start print {}".format(att))
    for i in range(Ndets):
        fig_i, ax_i = plt.subplots()
        x = data.time.values
        y = data[att][:, i].values
        ax_i.plot(x, y, "o-")

        ax_i.set_xlabel("time")
        ax_i.set_ylabel("channel")

        days = mdates.DayLocator()   # every day
        # months = mdates.MonthLocator()  # every month

        # days_fmt = mdates.DateFormatter('%d')
        major_fmt = mdates.DateFormatter('%d-%b')

        # format the ticks
        # ax_i.xaxis.set_major_locator(days)
        ax_i.xaxis.set_major_formatter(major_fmt)
        # ax_i.xaxis.set_minor_locator(days)
        ax_i.xaxis.set_major_locator(mticker.MaxNLocator(5))

        # format the coords message box
        ax_i.fmt_xdata = mdates.DateFormatter('%m-%d')

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig_i.autofmt_xdate()

        # delete from memory
        try:
            fig_i.savefig("figs/time/T_vs_{}_det{:2d}_time".format(att, i))
        except ValueError:
            print("Cannot plot att:{} det:{}".format(att, i))
        plt.close(fig_i)
    print("Done")


# def plotTandC1_1(data):


if __name__ == "__main__":
    # convert comma separation to dot separation
    os.system("sed -i 's/\,/\./g' data/run*.dat")

    if not os.path.exists("figs"):
        os.mkdir("figs")
    if not os.path.exists("figs/time"):
        os.mkdir("figs/time")
    if not os.path.exists("figs/run"):
        os.mkdir("figs/run")

    Ndets = 30

    fnames = getFiles("data")
    fnames = sortStringAndInt(fnames)
    time = []
    T = []
    data = []
    for fname in fnames:
        t_, T_, d_ = readDetData(fname)
        time.append(t_)
        T.append(T_)
        data.append(d_)

    data = xr.concat([df.to_xarray() for df in data], dim="dataset")
    data["T"] = T
    data["time"] = time
    # call like this:
    # a["c1"][runNo, DetNo]
    # a["c1"][runNo (starting at 0), DetNo (starting at 0)]

    # # some examples -- uncomment to run
    # # c1 of the second file, run02, and 6th detector (Detector5)
    # print("Ex1:", data["c1"][1, 5].values, "\n")
    # # c1 of the all runs, and 6th detector (Detector5)
    # print("Ex2:", data["c1"][:, 5].values, "\n")
    # # Temperatures of all runs
    # print("Ex3:", data.T.values, "\n")

    #print(data.time)
    plotAtts(data, "c1")
    #print(data.c1)
    
    plotAttsTime(data, "c1")
    plotAttsTime(data, "c2")



