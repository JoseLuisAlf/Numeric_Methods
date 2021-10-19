#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

#*********************  CONDICIONES DE FRONTERA **************************

def cond_frontera_dirichlet(a, b, T0, Tf, step, coef_vec, f, error):

    n_int = int((b-a)/step)
    n_pts = n_int + 1
    x = np.linspace(a, b, n_pts)
    dimM = n_pts - 2
    M = np.zeros((dimM, dimM))
    F = np.zeros(dimM)

    F[0] = f(x[1]) - coef_vec[0]*T0/pow(step, 2)
    F[dimM-1] = f(x[n_pts-2]) - coef_vec[2]*Tf/pow(step, 2)
    
    for i in range(1, dimM-1):
        F[i] = f(x[i+1])

    M[0][0] = coef_vec[1]/pow(step,2)
    M[0][1] = coef_vec[2]/pow(step,2)
    M[dimM-1][dimM-2] = coef_vec[0]/pow(step,2)
    M[dimM-1][dimM-1] = coef_vec[1]/pow(step,2)

    for i in range(0, n_pts-4):
        
        M[i+1][i+0] = coef_vec[0]/pow(step,2)
        M[i+1][i+1] = coef_vec[1]/pow(step,2)
        M[i+1][i+2] = coef_vec[2]/pow(step,2)
    
    T, errores, iteraciones = gauss_seidel(M,F,error)
    return T

#*************************************************************************

def cond_frontera_newman(a, b, alpha, Tf, step, coef_vec, f):

    n_int = int((b-a)/step)
    n_pts = n_int + 1
    x = np.linspace(a, b, n_pts)
    dimM = n_pts - 1
    M = np.zeros((dimM, dimM))
    F = np.zeros(dimM)

    F[0] = f(x[0]) + coef_vec[0]*2*alpha/step
    F[dimM-1] = f(x[n_pts-2]) - coef_vec[2]*Tf/pow(step, 2)
    
    for i in range(1, dimM-1):
        F[i] = f(x[i])

    M[0][0] = coef_vec[1]/pow(step,2)
    M[0][1] = (coef_vec[0]+coef_vec[2])/pow(step,2)
    M[dimM-1][dimM-2] = coef_vec[0]/pow(step,2)
    M[dimM-1][dimM-1] = coef_vec[1]/pow(step,2)

    for i in range(0, n_pts-3):
        
        M[i+1][i+0] = coef_vec[0]/pow(step,2)
        M[i+1][i+1] = coef_vec[1]/pow(step,2)
        M[i+1][i+2] = coef_vec[2]/pow(step,2)
    
    T = eliminacion_gauss(M, F)

    return T

#*************************************************************************

def metodo_potencias(A, errorf):

    dimA = len(A)
    v0 = np.ones(dimA) 
    error = 100
    eig_old = 0
    cont = 0
    while error > errorf:

        x = np.dot(A, v0)
        eig_value = x[0]
        error = abs((eig_value-eig_old)/eig_value)*100
        v0 = x/x[0]
        eig_old = eig_value
        cont = cont + 1
        if cont == 1000:
            error = 0

    return v0, eig_value, error, cont

#*********************************************************************

#************************   ELIMINACION DE GAUSS    ******************
def eliminacion_gauss(A,B):

    def pivoteo_parcial(A):

        n = len(A)
        M = np.copy(A)
        Naux=np.zeros(n)
        for i in range(n):
            if M[i][i] == 0:
                aux=M[i][i]
                for j in range(i+1,n):
                    if M[j][i]>aux:
                        aux=M[j][i]
                        index=j
                for l in range(n):
                    Naux[l]=M[index][l]
                    M[index][l]=M[i][l]
                    M[i][l]=Naux[l]
        return M

    def sust_inversa(A,B):

        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[n-1-i]=B[n-i-1]/A[n-i-1][n-i-1]
            for j in range(n-i,n):
                sol[n-1-i]=sol[n-1-i]-A[n-1-i][j]*sol[j]/A[n-1-i][n-1-i]

        return sol

    n = len(A)
    new_A=np.zeros((n,n+1))
    for i in range(n):
        new_A[i]=np.concatenate((A[i],[B[i]]))

    new_A = pivoteo_parcial(new_A)

    for i in range(0,n-1):
        for j in range(1,n-i):
            aux=[new_A[j+i][i]/new_A[i][i]*s for s in new_A[i]]
            #print(aux)
            for l in range(n+1):
                new_A[j+i][l]=new_A[j+i][l]-aux[l]

    new_B = np.zeros(n)
    for i in range(n):
        new_B[i] = new_A[i][n]

    new_A = np.delete(new_A, n, axis = 1)

    sol = sust_inversa(new_A, new_B)

    return sol
#**********************************************************************

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































