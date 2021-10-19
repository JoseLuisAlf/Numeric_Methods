#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *

#MODULOS: INTEGRACION

#**************** Polinomios de Lagrange ********************
def p_lagrange(x, y, v):

    def coef(x,v,i):
        l=1
        for j in range(len(x)):
            if j!= i:
                l=l*(v-x[j])/(x[i]-x[j])
        return l
    
    f=0
    for i in range(len(y)):
        f=f+y[i]*coef(x,v,i)
        
    return f
#************************************************************

#*************** Puntos de Chebyshev ************************
def puntos_chebyshev(a, b, num_puntos):

    p_cheby = np.zeros(num_puntos)
    for i in range(num_puntos):
        p_cheby[i] = a+(b-a)*0.5*(1+math.cos(i*math.pi/num_puntos))
    p_cheby = p_cheby[::-1]
    return p_cheby
#************************************************************

# *************** Método del trapecio compuesto ****************

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

# **************** Método de simpson (3/8) compuesto ***************
def met_simpson38_comp(a, b, n, funcion):

    if n%3 == 0:

        h = (b-a)/n
        x = np.zeros(n+1)
        for i in range(n+1):
            x[i] = a + i*h
        suma = 0
        for i in range(1, n):

            if i%3 != 0:
                suma = suma + 3*funcion(x[i])
            else:
                suma = suma + 2*funcion(x[i])

        integral = (3*h/8)*(funcion(x[0]) + suma + funcion(x[n]))
        
        return integral

    else:
        print('n debe ser multilpo de 3. Intente de nuevo.')

#*****************************************************************************

# MODULOS: EDO

#------------------------   ELIMINACION DE GAUUS    --------------------------
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
# ****************************** MÉTODO DE EULER ******************************

def euler_method(x0, y0, xf, step, equation):

    n = int((xf - x0)/step + 1)  # número de puntos
    x = np.linspace(x0, xf, n) # especificar el tipo de dato
    y = np.linspace(x0, xf, n)
    x[0], y[0] = x0, y0

    for i in range(1,n):
        y[i] = y[i-1] + equation(x[i-1], y[i-1])*step
    
    return x, y

# ****************************** MÉTODO DE HEUN *******************************

def heun_method(x0, y0, xf, step, equation, errorf):

    n = int((xf - x0)/step + 1)  # número de puntos
    x = np.linspace(x0, xf, n) # especificar el tipo de dato
    y = np.linspace(x0, xf, n)
    x[0], y[0] = x0, y0
    iteraciones = np.zeros(n-1)
    for i in range(n-1):
        
        error = 100
        predictor = y[i] + equation(x[i], y[i])*step
        
        while error > errorf:

            y[i+1] = y[i] + (equation(x[i], y[i]) + equation(x[i+1], predictor))*(step/2)
            error = abs((y[i+1] - predictor)/y[i+1])*100
            iteraciones[i] = iteraciones[i] + 1
            predictor = y[i+1]

    return x, y, iteraciones

# ****************************** MÉTODO DEL PUNTO MEDIO ****************************

def midpoint_method(x0, y0, xf, step, equation):

    n = int((xf - x0)/step + 1)  # número de puntos
    x = np.linspace(x0, xf, n) # especificar el tipo de dato
    y = np.linspace(x0, xf, n)
    x[0], y[0] = x0, y0

    for i in range(1, n):
        
        y_medio = y[i-1] + equation(x[i-1], y[i-1])*(step/2)
        y[i] = y[i-1] + equation(x[i-1] + step/2, y_medio)*step

    return x, y

# ****************************** RUNGE - KUTTA *************************************

def runge_kutta(x0, y0, xf, step, equation, order, a2):

    n = int((xf - x0)/step + 1)  # número de puntos
    x = np.linspace(x0, xf, n) # especificar el tipo de dato
    y = np.linspace(x0, xf, n)
    x[0], y[0] = x0, y0

    if order == 2:
        
        a1 = 1 - a2
        p1 = 1/(2*a2)
        q11 = p1
        
        for i in range(1, n):

            k1 = equation(x[i-1], y[i-1])
            k2 = equation(x[i-1] + p1*step, y[i-1] + q11*k1*step)

            y[i] = y[i-1] + (a1*k1 + a2*k2)*step

    if order == 3:

        for i in range(1, n):

            k1 = equation(x[i-1], y[i-1])
            k2 = equation(x[i-1] + 0.5*step, y[i-1] + 0.5*k1*step)
            k3 = equation(x[i-1] + step, y[i-1] - k1*step + 2*k2*step)

            y[i] = y[i-1] + (1/6)*(k1 + 4*k2 + k3)*step

    if order == 4:

        for i in range(1, n):

            k1 = equation(x[i-1], y[i-1])
            k2 = equation(x[i-1] + 0.5*step, y[i-1] + 0.5*k1*step)
            k3 = equation(x[i-1] + 0.5*step, y[i-1] + 0.5*k2*step)
            k4 = equation(x[i] + step, y[i] + k3*step)

            y[i] = y[i-1] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*step

    if order == 5:

        for i in range(1, n):

            k1 = equation(x[i-1], y[i-1])
            k2 = equation(x[i-1] + 0.25*step, y[i-1] + 0.25*k1*step)
            k3 = equation(x[i-1] + 0.25*step, y[i-1] + 0.125*k1*step + 0.125*k2*step)
            k4 = equation(x[i-1] +0.5*step, y[i-1] - 0.5*k2*step + k3*step)
            k5 = equation(x[i-1] + 0.75*step, y[i-1] + (3/16)*k1*step + (9/16)*k4*step)
            k6 = equation(x[i-1] + step, y[i-1] - (3/7)*k1*step + (2/7)*k2*step + (12/7)*k3*step - (12/7)*k4*step + (8/7)*k5*step)

            y[i] = y[i-1] + (1/90)*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)*step

    return x, y
            
#****************************** SISTEMA DE ECUACIONES *****************************

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

#******************************** EULER EXPLICITO ***************************

def s_ode_eulerexp(x0, xf, y0_vec, step, equ_vector): #, metodo):

    n_int = int((xf - x0)/step)
    n_pts = int(n_int + 1)
    n_eq = len(equ_vector)
    r = np.zeros((n_pts, n_eq))
    x = np.linspace(x0, xf, n_pts) # especificar el tipo de dato
    r[0] = y0_vec 
   
    for i in range(n_int):
        
        for j in range(n_eq):

            r[i+1][j] = r[i][j] + equ_vector[j](x[i], r[i])*step

    return r
        
#******************************** EULER IMPLICITO ***************************

def s_ode_eulerimp(x0, xf, y0_vec, step, coef1, coef2):#, metodo):

    n_int = int((xf - x0)/step)
    n_pts = int(n_int + 1)
    n_eq = len(y0_vec)
    r = np.zeros((n_pts, n_eq))
    x = np.linspace(x0, xf, n_pts) # especificar el tipo de dato
    r[0] = y0_vec 
    M = np.array([[-1+coef1[0]*step, coef1[1]*step], \
                  [coef2[0]*step, -1 + coef2[1]*step]])
    for i in range(n_int):
        
        r[i+1] = eliminacion_gauss(M, -1*r[i])

    return r

#****************************** HEUN MODIFICADO ****************************

def heun_modify(x0, xf, y0, yb, step, equation, errorf):

    n_int = int((xf - x0)/step) # número de intervalos
    n_pts = int(n_int +1)
    x = np.linspace(x0, xf, n_pts) # especificar el tipo de dato
    y = np.zeros(n_pts)
    x = np.insert(x, 0 , x0 - step)
    y = np.insert(y, 0, yb)
    x[1], y[1] = x0, y0
    iteraciones = np.zeros(n_pts-1)

    for i in range(1, n_pts):
        
        error = 0
        predictor = y[i-1] + equation(x[i], y[i])*(2*step)
        
        while error > errorf or error == 0:

            y[i+1] = y[i] + (equation(x[i], y[i]) + equation(x[i+1], predictor))*(step/2)
            error = abs((y[i+1] - predictor)/y[i+1])*100
            iteraciones[i-1] = iteraciones[i-1] + 1
            predictor = y[i+1]

    return x, y, iteraciones


















