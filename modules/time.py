#!/usr/bin/env python

import time

###############################################################################
# global variables
###############################################################################
t0=-1
reinit=1

###############################################################################
# functions
###############################################################################
def init():
    global t0
    t0=time.time()

def elapsed(s):
    global t0
    global reinit
    if t0==-1:
        raise Exception("ERROR: time not initialized")
    t1=time.time()
    t=t1-t0
    if s:
        print("%s: %f" % (s,t))
    else:
        print("Time: %f" % t)
    if reinit:
        t0=t1

