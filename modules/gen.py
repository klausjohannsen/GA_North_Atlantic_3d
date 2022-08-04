#!/usr/bin/env python

# std libs
import sys
import numpy as np
import numpy.linalg as la
import sklearn as skl
import scipy.spatial.distance as ssd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import itertools

###############################################################################
# functions
###############################################################################
def dist(X):
    d=ssd.pdist(X)
    return(np.percentile(d,0),np.percentile(d,25),np.percentile(d,50),np.percentile(d,75),np.percentile(d,100))

def tensorgrid(min,max,num):
    x=np.linspace(min[0],max[0],num=num[0])
    y=np.linspace(min[1],max[1],num=num[1])
    xx,yy=np.meshgrid(x,y)
    Z=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
    return(Z)

def gnuwrite(file,X):
    with open(file,'w') as out:
        old=X[0,1]
        for i in range(X.shape[0]):
            if old!=X[i,1]:
                out.write("\n")
                old=X[i,1]
            for j in range(X.shape[1]):
                out.write(str(X[i,j])+' ')
            out.write("\n")
                


