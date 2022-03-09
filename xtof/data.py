import os
import time
import datetime
import matplotlib.pyplot as plt
import collections

"""
papers:
    - https://arxiv.org/pdf/2005.07057v2.pdf = estime le WEAR pas le RUL
    - estime une accuracy de wear ? Fault Diagnosis from Raw Sensor Data Using Deep Neural Networks Considering Temporal Coherence

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

