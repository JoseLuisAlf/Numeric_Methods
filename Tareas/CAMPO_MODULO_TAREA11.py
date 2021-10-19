#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

#************************** ECUACIONES ELÍPTICAS ******************************

def equ_eliptica(x0, xf, npt, f, alpha1, alpha2, Tx0, Ty0, x, y, errorf):

    step = (x0-xf)/(npt-1)

    T = np.zeros((npt, npt))
    T[0] = Tx0
    for i in range(npt):
        T[i][npt-1] = Ty0[npt-1-i]
    
    dimM = (npt-1)**2
    M = np.zeros((dimM,dimM))
    F = np.zeros(dimM)

    # Armamos la matriz

    #Primera fila
    M[0][0] = -4
    M[0][1] = 2
    M[0][npt-1] = 2

    #
    for i in range(npt-3):
        M[i+1][i+0] = 1
        M[i+1][i+1] = -4
        M[i+1][i+2] = 1
        M[i+1][i+npt] = 2

    #
    M[npt-2][npt-3] = 1
    M[npt-2][npt-2] = -4
    M[npt-2][npt-2 + (npt-1)] = 2

    #
    for i in range(npt-3):

        M[(npt-1)*(i+1)][(npt-1)*i] = 1
        M[(npt-1)*(i+1)][(npt-1)*(i+1)] = -4
        M[(npt-1)*(i+1)][(npt-1)*(i+1)+1] = 2
        M[(npt-1)*(i+1)][(npt-1)*(i+2)] = 1

    #
    for j in range(npt-3):

        for i in range(npt-3):
        
            M[npt+(npt-1)*i+j][1 + (npt-1)*i + j] = 1
            M[npt+(npt-1)*i+j][(npt-1)*(i+1) + j] = 1
            M[npt+(npt-1)*i+j][(npt-1)*(i+1)+1 + j] = -4
            M[npt+(npt-1)*i+j][(npt-1)*(i+1)+2 + j] = 1
            M[npt+(npt-1)*i+j][(npt-1)*(i+1)+npt + j] = 1
    
    #
    for i in range(npt-3):

        M[(npt-1)*(i+2)-1][npt-2 + (npt-1)*i] = 1
        M[(npt-1)*(i+2)-1][2*(npt-2)+ (npt-1)*i] = 1
        M[(npt-1)*(i+2)-1][2*(npt-2)+ (npt-1)*i+1] = -4
        M[(npt-1)*(i+2)-1][2*(npt-2)+ (npt-1)*(i+1) +1] = 1
    #
    M[(npt-1)*(npt-2)][(npt-2)**2-1] = 1
    M[(npt-1)*(npt-2)][(npt-2)**2+npt-2] = -4
    M[(npt-1)*(npt-2)][(npt-2)**2+npt-1] = 2
    
    #
    for i in range(npt-3):
        
        M[(npt-1)*(npt-2)+(i+1)][(npt-2)**2+i] = 1
        M[(npt-1)*(npt-2)+(i+1)][(npt-2)**2+i + (npt-2)] = 1
        M[(npt-1)*(npt-2)+(i+1)][(npt-2)**2+i + (npt-1)] = -4
        M[(npt-1)*(npt-2)+(i+1)][(npt-2)**2+i + npt] = 1


    M[npt*(npt-2)][(npt-2)**2+(npt-3)] = 1
    M[npt*(npt-2)][(npt-2)**2+2*npt-5] = 1
    M[npt*(npt-2)][(npt-2)*npt] = -4

    M = (1/pow(step,2))*M

    # Ahora formamos la matriz F

    F[0] = f(x[0],y[0]) + 2*alpha1/step + 2*alpha2/step
    F[dimM-1] = f(x[npt-2],y[npt-2]) - T[1][npt-1]/step**2 - T[0][npt-2]/step**2
    F[npt-2] = f(x[npt-2],y[0]) + 2*alpha2/step-T[npt-1][npt-1]/step**2
    F[dimM-4] = f(x[0], y[npt-2]) +2*alpha1/step-T[0][0]/step**2
    
    for i in range(npt-3):

        F[i+1] = f(x[i+1], y[0]) + 2*alpha2/step

    for i in range(npt-3):

        F[(npt-1)*(i+1)] = f(x[0],y[i+1]) + 2*alpha1/step

    for i in range(npt-3):
        for j in range(npt-3):

            F[npt + i + (npt-1)*j] = f(x[i+1],y[j+1])

    for i in range(npt-3):

        F[2*npt-3 + (npt-1)*i] = f(x[npt-2], y[i+1]) - T[npt-2-i][npt-1]/step**2

    for i in range(npt-3):

        F[(npt-1)*(npt-2)+i+1] = f(x[i+1], y[npt-1]) - T[0][i+1]/step**0

    # resolvemos
    Taux, error, iteraciones = gauss_seidel(M,F,errorf)     
        
    return T, Taux

#************************ Metodo Gauss - Seidel **************************
def gauss_seidel(A, B, accurate):
    
    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    size = len(A)
    iterations=0
    error = np.array([])
    x=np.zeros(size)
    xold=np.zeros(size)
    bolean=True
    while bolean :
        for i in range(size):
            xold[i]=x[i]
        for i in range(size):
            x[i]=B[i]/A[i][i]
            for j in range(i+1,size):
                x[i]=x[i]-A[i][j]*x[j]/A[i][i]
            for j in range(0,i):
                x[i]=x[i]-A[i][j]*x[j]/A[i][i]
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold) < accurate:
            bolean=False
        
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones

#*************************************************************************
