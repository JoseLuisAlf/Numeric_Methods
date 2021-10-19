#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

# **************************** CRANK - NICOLSON *********************************

def crank_nicolson(a, b, Fa, Fb, n_pts, step_time, n_time, T0, alpha, errorf):

    def g(x, T, s):

        g = 0.5*s*T[x-1] + (1-s)*T[x] + 0.5*s*T[x+1]
        return g

    step_x = (b-a)/(n_pts-1)
    xdom   = np.linspace(a, b, n_pts) 
    npx_int = n_pts - 2
    s = alpha*step_time/pow(step_x,2)
    M = np.zeros((npx_int, npx_int))
    T = np.zeros((n_time, n_pts))

    T[0] = T0

    M[0][0] = 1+s
    M[0][1] = -0.5*s
    M[npx_int-1][npx_int-2] = -0.5*s
    M[npx_int-1][npx_int-1] = 1+s

    for i in range(1, npx_int-1):

        M[i][i-1] = -0.5*s
        M[i][i]   = 1+s
        M[i][i+1] = -0.5*s

    F = np.zeros(npx_int)

    for i in range(1, n_time):
        
        F[0] = g(1, T[i-1], s) + 0.5*s*Fa(i*step_time)
        F[npx_int-1] = g(npx_int, T[i-1], s) + 0.5*s*Fb(i*step_time)
        
        for j in range(2, npx_int):

            F[j-1] = g(j, T[i-1], s)
        #print(F)
        #print(M)
        Taux, error, iteraciones = gauss_seidel(M, F, errorf)
        Taux = np.insert(Taux, 0, Fa(i*step_time))
        Taux = np.insert(Taux, npx_int+1, Fb(i*step_time))
        T[i] = Taux

    return T, xdom
#*************************************************************************  

#*************************************** RK4 *****************************
def s_ode_RK4(x0, xf, y0_vec, step, equ_vector): #, metodo):

    n_int = int((xf - x0)/step)
    n_pts = int(n_int + 1)
    n_eq = len(equ_vector)
    r = np.zeros((n_pts, n_eq))
    k1 = np.zeros(n_eq)
    k2 = np.zeros(n_eq)
    k3 = np.zeros(n_eq)
    k4 = np.zeros(n_eq)
    x = np.linspace(x0, xf, n_pts) # especificar el tipo de dato
    r[0] = y0_vec 
    for i in range(n_int):

        for j in range(n_eq):

            k1[j] = equ_vector[j](x[i], r[i])
            #print('k1{}: '.format(j+1), k1[j]) 
        for j in range(n_eq):

            k2[j] = equ_vector[j](x[i] + 0.5*step, r[i] + 0.5*k1*step)
            #print('k2{}: '.format(j+1), k2[j]) 
        
        for j in range(n_eq):

            k3[j] = equ_vector[j](x[i] + 0.5*step, r[i] + 0.5*k2*step)
            #print('k3{}: '.format(j+1), k3[j]) 

        for j in range(n_eq):

            k4[j] = equ_vector[j](x[i] + step, r[i] + k3*step)
            #print('k4{}: '.format(j+1), k4[j]) 

        for j in range(n_eq):

            r[i+1][j] = r[i][j] +(1/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])*step

    return r

#*************************************************************************

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
