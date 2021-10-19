#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

# ************************** INTEGRACION NUMERICA I ************************

# ******************** Método del trapecio *********************
def met_trapecio_p(x0, x1, f0, f1):
    
    integral = (x1 - x0)*(f0 + f1)/2
    return integral

def met_trapecio_f(x0, x1, funcion):

    integral = (x1 - x0)*(funcion(x0) + funcion(x1))/2
    return integral

# *************** Método del trapecio compuesto ****************
def met_trapecio_compuesto_p(x, y):

    n = len(x) - 1
    suma = 0
    for i in range(1, n):
        suma = suma + y[i]
    integral = (x[n]-x[0])*(y[0] + 2*suma + y[n])/(2*n)
    return integral

def met_trapecio_compuesto_f(x, funcion):

    n = len(x) - 1
    suma = 0
    for i in range(1, n):
        suma = suma + funcion(x[i])
    integral = (x[n]-x[0])*(funcion(x[0]) + 2*suma + funcion([n]))/(2*n)
    return integral

def trapecio_com_fn(a,b, n, funcion):

    h = (b-a)/n
    x = np.zeros(n+1)
    for i in range(n+1):
        x[i] = a + i*h
    suma = 0
    for i in range(1, n):
        suma = suma + funcion(x[i])
    integral = (x[n]-x[0])*(funcion(x[0]) + 2*suma + funcion(x[n]))/(2*n)
    return integral

# **************** Método de Simpson (1/3) *********************
def met_simpson13_p(x0, x2, y0, y1, y2):

    x1 = (x0 + x2)/2
    integral = (x2 - x0)*(y0 + 4*y1 + y2)/6
    return integral

def met_simpson13_f(x0, x2, funcion):

    x1 = (x0 + x2)/2
    integral = (x2 - x0)*(funcion(x0) + 4*funcion(x1) + funcion(x2))/6
    return integral

# ************** Método de simpson compuesto (1/3) *************
def met_simpson13_compuesto_p(x, y):

    n = len(x) - 1
    suma_odd = 0
    suma_even = 0
    for i in range(1, n):
        if i%2 != 0:
            suma_odd = suma_odd + y[i]
        else:
            suma_even = suma_even + y[i]

    integral = (x[n] - x[0])*(y[0] + 4*suma_odd + 2*suma_even + y[n])/(3*n)
    return integral

def met_simpson13_compuesto_f(x, funcion):

    n = len(x) - 1
    suma_odd = 0
    suma_even = 0
    for i in range(1, n):
        if i%2 != 0:
            suma_odd = suma_odd + funcion(x[i])
        else:
            suma_even = suma_even + funcion(x[i])

    integral = (x[n] - x[0])*(funcion(x[0]) + 4*suma_odd + 2*suma_even + funcion([n]))/(3*n)
    return integral

# ***************** Método de simpson (3/8) ***********************
def met_simpson38_p(x, y):

    n = len(x) - 1
    integral = (x[n] - x[0])*(y[0] + 3*y[1] + 3*y[2] + y[3])/8
    return integral

def met_simpson38_f(x, funcion):

    n = len(x) - 1
    integral = (x[n] - x[0])*(funcion(x[0]) + 3*funcion(x[1]) + 3*funcion(x[2]) + funcion(x[3]))/8
    return integral

#****************** Extrapolación de Richardson ********************
def extrapolacion_richardson(a, b, n1, n2, funcion):


    h1 = (b-a)/n1
    h2 = (b-a)/n2

    new_integral = trapecio_com_fn(a, b, n2, funcion) + \
                    (trapecio_com_fn(a, b, n2, funcion) - \
                    trapecio_com_fn(a, b, n1, funcion))/(math.pow(h1/h2, 2) - 1)
                    
    return new_integral

# ************************* Trapecio modificado ********************
def trapecio_no_equi(x,y):
    
    n = len(x)-1
    h = np.zeros(n)
    for i in range(n):
        h[i]= x[i+1] - x[i]
    integral = 0
    for i in range(n):
        integral = integral + h[i]*(y[i] + y[i+1])/2
    return integral

# ************************** INTEGRACION NUMERICA II ************************

# ********************* GAUSS - LEGENDRE**********************************

def gauss_legendre(x0, xf, equation, puntos):

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
        
        suma = suma + w_aux[i]*equation( (xf - x0)*x_aux[i]/2  + (x0 + xf)/2 )

    suma = (xf - x0)*suma/2

    return suma

# ********************* GAUSS - RADAU ************************************

def gauss_radau(x0, xf, equation, puntos):

    x3 = np.array([-1, -0.289898, 0.689898])
    x4 = np.array([-1, -0.575319, 0.181066, 0.822824])
    x5 = np.array([-1, -0.720480, -0.167181, 0.446314, 0.885792])

    w3 = np.array([0.222222, 1.0249717, 0.7528061])
    w4 = np.array([0.125000, 0.657689, 0.776387, 0.440924])
    w5 = np.array([0.080000, 0.446208, 0.623653, 0.562712, 0.287427])

    raiz = {'x3': x3, 'x4': x4, 'x5': x5}
    peso = {'w3': w3, 'w4': w4, 'w5': w5}
    x_aux = raiz['x{}'.format(puntos)] # Definir la instruccin para asignar
    w_aux = peso['w{}'.format(puntos)]
    suma = 0
    for i in range(len(x_aux)):
        
        suma = suma + w_aux[i]*equation( (xf - x0)*x_aux[i]/2  + (x0 + xf)/2 )

    suma = (xf - x0)*suma/2

    return suma

# ********************* GAUSS - LOBATTO **********************************

def gauss_lobatto(x0, xf, equation, puntos):

    x2 = np.array([-1, 1])
    x3 = np.array([-1, 0, 1])
    x4 = np.array([-1, -0.447213, 0.447213, 1])
    x5 = np.array([-1, -0.654653, 0, 0.654653, 1])
    x6 = np.array([-1, -0.765055, -0.285231, 0.285231, 0.765055, 1])
    x7 = np.array([-1, -0.830223, -0.468848, 0, 0.468848, 0.830223, 1])
    x8 = np.array([-1, -0.871740, -0.591700, -0.209299, 0.209299, 0.591700, 0.871740, 1])

    w2 = np.array([1,1])
    w3 = np.array([0.333333, 1.333333, 0.333333])
    w4 = np.array([0.166666, 0.833333, 0.833333, 0.166666])
    w5 = np.array([0.1, 0.544444, 0.711111, 0.544444, 0.1])
    w6 = np.array([0.066666, 0.378474, 0.554858, 0.554858, 0.378474, 0.066666])
    w7 = np.array([0.047619, 0.276826, 0.431745, 0.487619, 0.431745, 0.276826, 0.047619])
    w8 = np.array([0.035714, 0.210704, 0.341122, 0.412458, 0.412458, 0.341122, 0.210704, 0.035714])

    raiz = {'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8}
    peso = {'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8}
    x_aux = raiz['x{}'.format(puntos)] # Definir la instruccin para asignar
    w_aux = peso['w{}'.format(puntos)]
    suma = 0
    for i in range(len(x_aux)):
        
        suma = suma + w_aux[i]*equation( (xf - x0)*x_aux[i]/2  + (x0 + xf)/2 )

    suma = (xf - x0)*suma/2

    return suma

# *************************** DIFERENCIACION *****************************

def dif_forward(y, step, derivada, i):

    if derivada == 1:

        f = (-y[i+2] +4*y[i+1] - 3*y[i])/(2*step)
        return f

    if derivada == 2:

        f = (-y[i+3] + 4*y[i+2] - 5*y[i+1] + 2*y[i])/math.pow(step, 2)
        return f

def dif_backward(y, step, derivada, i):

    if derivada == 1:

        f = (3*y[i] -4*y[i-1] + y[i-2])/(2*step)
        return f

    if derivada == 2:

        f = (2*y[i] - 5*y[i-1] + 4*y[i-2] - y[i-3])/math.pow(step, 2)
        return f

def dif_centrada(y, step, derivada, i):


    if derivada == 1:

        f = (y[i+1] - y[i-1])/(2*step)
        return f

    if derivada == 2:

        f = (y[i+1] - 2*y[i] + y[i-1])/(math.pow(step, 2))
        return f

# ************************************************************************

def met_forward(y, step, derivada):
    
    n = len(y)
    m = derivada + 1
    y_for = np.zeros(n)
    
    for i in range(0, n - m):
        y_for[i] = dif_forward(y, step, derivada, i)
    return y_for
            
def met_backward(y, step, derivada):
    
    n = len(y)
    m = derivada + 1
    y_for = np.zeros(n)

    for i in range(m, n):
        y_for[i] = dif_backward(y, step, derivada, i)
    return y_for

def met_centrada(y, step, derivada):

    n = len(y)
    m = 1
    y_for = np.zeros(n)

    for i in range(m, n - m):
        y_for[i] = dif_centrada(y, step, derivada, i)
    return y_for
