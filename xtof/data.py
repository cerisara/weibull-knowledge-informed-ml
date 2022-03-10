import os
import time
import datetime
import matplotlib.pyplot as plt
import collections

"""
papers:
    - https://arxiv.org/pdf/2005.07057v2.pdf = estime le WEAR pas le RUL
    - estime une accuracy de wear ? Fault Diagnosis from Raw Sensor Data Using Deep Neural Networks Considering Temporal Coherence

approche choisie = estimation du RUL: la doc dit "test to failure experiment", donc je suppose que la fin de la
serie correspond a la fin de vie d'un des bearing.

Test1: bearings 3 et 4 ont une failure
Test2: bearing 1 a une failure
Test3: bearing 3 a une failure

"""

ddir = "../data/raw/IMS/"

def loadTrain():
    trdir = ddir+"2nd_test/"
    date_list = sorted(os.listdir(trdir))
    col_names = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    start_time= date_list[0]
    start_time = time.mktime(
        datetime.datetime.strptime(start_time, "%Y.%m.%d.%H.%M.%S").timetuple()
    )
    allseqs = []
    allruls = []
    vmin,vmax = 99999.,-99999.
    for i, sample_name in enumerate(date_list):
        t = time.mktime(datetime.datetime.strptime(sample_name, "%Y.%m.%d.%H.%M.%S").timetuple())
        onesecseq = []
        with open(trdir+sample_name,"r") as f:
            for l in f:
                # on ne garde que bearing 1, car c'est le seul a avoir une failure
                v = float(l.split("\t")[0])
                if v<vmin: vmin=v
                elif v>vmax: vmax=v
                onesecseq.append(v)
        allseqs.append(onesecseq)
        allruls.append(t)
    lastT = allruls[-1]
    allruls = [lastT-x for x in allruls]
    largestRUL = allruls[0]
    allruls = [x/largestRUL for x in allruls]
    res=[]
    vmax -= vmin
    for s in allseqs:
        res.append([(x-vmin)/vmax for x in s])
    return res,allruls
 
def showTrain():
    trdir = ddir+"2nd_test/"
    date_list = sorted(os.listdir(trdir))
    col_names = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    start_time= date_list[0]
    start_time = time.mktime(
        datetime.datetime.strptime(start_time, "%Y.%m.%d.%H.%M.%S").timetuple()
    )
    d = collections.deque(maxlen=100)
    for i, sample_name in enumerate(date_list):
        unix_timestamp = time.mktime(
            datetime.datetime.strptime(sample_name, "%Y.%m.%d.%H.%M.%S").timetuple()
        )
        date_nice_format = datetime.datetime.fromtimestamp(unix_timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        with open(trdir+sample_name,"r") as f:
            for l in f:
                d.append(l)
    yy0=[]
    yy1=[]
    yy2=[]
    yy3=[]
    for l in d:
        v = l.split("\t")
        yy0.append(float(v[0]))
        yy1.append(float(v[1]))
        yy2.append(float(v[2]))
        yy3.append(float(v[3]))
    plt.plot(yy0,label="b1")
    plt.plot(yy1,label="b2")
    plt.plot(yy2,label="b3")
    plt.plot(yy3,label="b4")
    plt.legend()
    plt.show()

loadTrain()

