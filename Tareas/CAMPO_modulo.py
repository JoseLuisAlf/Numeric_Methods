#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

# ********************* GAUSS - LEGENDRE**********************************

def gauss_legendre(x0, xf, equation, parametro, puntos):

    x1 = np.array([0])
    x2 = np.array([-0.57735, 0.57735])
    x3 = np.array([-0.774597, 0, 0.774597])
    x4 = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
    x5 = np.array([-0.90618, -0.538469, 0, 0.538469, 0.90618])

    w1 = np.array([2])
    w2 = np.array([1,1])
    w3 = np.array([5/9, 8/9, 5/9])
    w4 = np.array([0.347855, 0.652145, 0.652145, 0.347855])
    w5 = np.array([0.236927, 0.478629, 0.568889, 0.478629, 0.236927])

    raiz = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}
    peso = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5}
    x_aux = raiz['x{}'.format(puntos)] # Definir la instruccin para asignar
    w_aux = peso['w{}'.format(puntos)]
    suma = 0
    for i in range(len(x_aux)):
        
        suma = suma + w_aux[i]*equation( (xf - x0)*x_aux[i]/2  + (x0 + xf)/2 , parametro)

    suma = (xf - x0)*suma/2

    return suma

#*************************************** RK4 ***********************************
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

#************************ Ecuaciones en derivadas parciales ************************

def equ_parabolica(a, b, fact, n_pts, step_t, t_lim, f, x0, xf, parametro, error):

    stepx = ((b-a)/(n_pts-1))
    np_int = n_pts - 2
    x = np.linspace(a, b, n_pts)
    n = np.zeros((t_lim+1, n_pts))
    M = np.zeros((np_int, np_int))
    F = np.zeros(np_int)

    for i in range(t_lim+1): # Condiciones de frontera

        n[i][0] = x0[i]
        n[i][n_pts-1] = xf[i]

    for i in range(np_int+2): # Definimos los valores iniciales

        n[0][i] = f(x[i], parametro) 

    # Construimos la matriz
    M[0][0] = fact[2]
    M[0][1] = -1*fact[0]
    M[np_int-1][np_int-2] = -1*fact[0]
    M[np_int-1][np_int-1] = fact[2]

    for i in range(0, np_int - 2):
        
        M[i+1][i+0] = -1*fact[0]
        M[i+1][i+1] = fact[2]
        M[i+1][i+2] = -1*fact[0]

    # Calculamos el vector F
    for i in range(1, t_lim+1):

        F[0] = fact[0]*(n[i-1][0]+n[i][0])+fact[1]*n[i-1][1]+fact[0]*n[i-1][2]
        F[np_int-1] = fact[0]*n[i-1][np_int-1] + fact[1]*n[i-1][np_int] + fact[0]*(n[i-1][np_int+1]+n[i][np_int+1])
    
        for j in range(1, np_int-1):
            F[j] = fact[0]*n[i-1][j] + fact[1]*n[i-1][j+1] + fact[0]*n[i-1][j+2]

        # Con M y F construidos, resolvemos el sistema 
        N, errores, iteraciones = gauss_seidel(M,F,error)
        
        # Llenamos la matriz con los valores de n al tiempo respectivo
        for k in range(1, np_int+1):
            
            n[i][k] = N[k-1]
    
    return n

#************************************ Módulo: Regresión lineal ***********************
def XYdot_sum(x ,y):

    result = 0
    for i in range(len(x)):
        result = result + x[i] * y[i]
    return result

def XXdot_sum(x):
    result = 0
    for i in range(len(x)):
        result = result + math.pow(x[i], 2)
    return result

def vector_sum(x):
    result = 0
    for i in range(len(x)):
        result = result + x[i]
    return result

def vector_average(x):
    
    result = vector_sum(x) / len(x)
    return result
        
def linear_regression(x, y):

    coef0, coef1 = 0, 0
    coef1 = ( len(x) * XYdot_sum(x, y) - vector_sum(x) * vector_sum(y) )/ \
         ( len(x) * XXdot_sum(x) - math.pow(vector_sum(x), 2) )
    coef0 = vector_average(y) - coef1 * vector_average(x)
    return coef0, coef1

#**************************************************************************************
