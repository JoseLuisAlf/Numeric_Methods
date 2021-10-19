#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

#******************************* PREGUNTA1 ******************************

# FTCS
def FTCS_neumann(a, b, npt, time_step, n_time, T0, c, alplha):

    stepx = (b-a)/(npt-1)
    s = alpha*time_step/pow(stepx,2)

    T = np.zeros((n_time, npt))
    T[0] = T0

    for i in range(n_time-1):

        T[i+1][0] = -2*s*stepx*c((i+1)*time_step) + (1-2*s)*T[i][0] + 2*s*T[i][1] 

        for j in range(npt-2):

            T[i+1][j+1] = s*T[i][j] + (1-2*s)*T[i][j+1] + s*T[i][j+2]

    for i in range(n_time-1):

        T[i+1][npt-1] = f((i+1)*time_step)

    return T

#definimos el problema

a, b, npt, time_step, n_time, alpha = 0, 1, 6, 0.01, 10, 1
xdom = np.linspace(a,b,npt)
T0 = np.zeros(npt)

def c(t):

    c = 0.5
    return c

def fx(x):

    fx = 0
    return fx

def f(t):

    f = 1
    return f

for i in range(npt):
    T0[i] = fx(xdom[i])


T = FTCS_neumann(a, b, npt, time_step, n_time, T0, c, alpha)
print('\n')
print(T)

#Crank-Nicolson



