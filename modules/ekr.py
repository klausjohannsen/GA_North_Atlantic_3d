#!/usr/bin/env python

# std libs
import sys
import numpy as np
import numpy.linalg as la
import sklearn as skl
import scipy.spatial.distance as ssd
from sklearn.kernel_ridge import KernelRidge

# general modules
import modules.gen as gen

###############################################################################
# classes
###############################################################################
class fct:
    def __init__(self, X, Y, alpha=0.001, h=0, kernel='rbf'):

        # store
        self._X=X
        self._Y=Y
        self.n=Y.shape[1]
        self.m=X.shape[1]
        self._alpha=alpha

        # h
        self._h=gen.dist(X)[2]
        if h:
            self._h=h*self._h

        # fit
        self._ekr=KernelRidge(alpha=alpha, kernel=kernel, gamma=1./self._h/self._h)
        self._ekr.fit(X,Y)

    def __str__(self):
        s=''
        s+="%d <-- %d\n" % (self.n,self.m)
        s+="X: (%d, %d)\n" % self._X.shape
        s+="Y: (%d, %d)\n" % self._Y.shape
        s+="alpha=%e\n" % self._alpha
        s+="h=%e\n" % self._h
        return(s)

    def eval(self,X):
        Y=self._ekr.predict(X)
        return(Y)




