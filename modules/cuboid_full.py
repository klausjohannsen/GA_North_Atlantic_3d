#!/usr/bin/env python

# std libs
import numpy as np
import random
import numpy.linalg as la
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import modules.problem as problem
import modules.data as data
import time
import copy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.kernel_ridge import KernelRidge as KR
from scipy.spatial.distance import cdist

###############################################################################
# GA_simple_float
###############################################################################
class Cuboid(problem.NadineGoris_1):
    def __init__(self, area=[], hof=1, models=[], r=[]):
        super(Cuboid,self).__init__(Q='2090s', models=models)

        # store settings
        self._area=area
        self._hof=hof
        if hof!=1:
            raise Exception("ERROR: hof must be 1")
        self._neval=0
        self._m=np.full(self.n(), False, dtype=bool)
        self._M=data.mask(['m',self,self._m])
        self._r=r

        # build tensor product mesh
        p=self.points()
        cnt=0
        x=np.unique(p[:,0])
        for i in range(x.shape[0]):
            for ii in range(i,x.shape[0]):
                print("%d %d of %d %d" % (i,ii,len(x),len(x)))
                idx=(p[:,0]>=x[i])*(p[:,0]<=p[ii,0])
                pp=p[idx,:]
                y=np.unique(pp[:,1]) 
                for j in range(y.shape[0]):
                    for jj in range(j,y.shape[0]):
                        idx=(pp[:,1]>=pp[j,1])*(pp[:,1]<=pp[jj,1])
                        ppp=pp[idx,:]
                        z=np.unique(ppp[idx,2]) 
                        for k in range(z.shape[0]):
                            for kk in range(k,z.shape[0]):
                                cnt+=1
        print("cnt=%d" % cnt)

        # scan all possibilities
        p=self.points()
        print(p.shape)
        n=p.shape[0]
        self._cmax=self._amax=self._imax=self._jmax=-1
        cnt=0
        for i in range(1000):
            ix=p[:,0]>=p[i,0]
            iy=p[:,1]>=p[i,1]
            iz=p[:,2]>=p[i,2]
            idx=ix*iy*iz
            for j in np.where(idx)[0]:
                mx=(p[:,0]>=p[i,0])*(p[:,0]<=p[j,0])
                my=(p[:,1]>=p[i,1])*(p[:,1]<=p[j,1])
                mz=(p[:,2]>=p[i,2])*(p[:,2]<=p[j,2])
                m=mx*my*mz
                self._M.set_mask(m)
                self._neval+=1
                (c,a)=self.cost(self._M)
                if a>=self._area[0] and a<=self._area[1]:
                    if c>self._cmax:
                        self._cmax=c
                        self._amax=a
                        self._imax=i
                        self._jmax=j
            cnt+=1
            print("%d: c=%f, a=%f" % (cnt,self._cmax,self._amax))

    def hof(self):
        return([(self._cmax,self._amax)])

    def gof(self):
        return([(self._imax,self._jmax)])

    def neval(self):
        return(self._neval)

    def __str__(self):
        s="--- Cuboid ---\n"
        s=s+"area=[%.2f, %.2f]\n" % (self._area[0],self._area[1]);
        s=s+"hof=%d\n" % (self._hof);
        s=s+"\n"
        s=s+super(Cuboid,self).__str__()

        return(s)







