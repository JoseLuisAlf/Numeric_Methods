#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from CAMPO_MODULO_TAREA8 import *

print('*********************** INTEGRACION NUMERICA I *************************')

# ************************* PREGUNTA 1 ******************************
print('PREGUNTA 1: ')
print('\n')

def funcion1(v):
    f = 1 - v - 4*math.pow(v, 3) + 2*math.pow(v, 5)
    return f

a = -2
b = 4
x = np.zeros(4)
for i in range(4):
    x[i] = a +i*(b-a)/3
n1 = 1
n2 = 100

valor_analitico = 1104
trapecio = met_trapecio_f(a, b, funcion1)
trapecio_comp1 = trapecio_com_fn(a, b, n1, funcion1)
trapecio_comp2 = trapecio_com_fn(a, b, n2, funcion1)
simpson_13 = met_simpson13_f(a, b, funcion1)
simpson_38 = met_simpson38_f(x, funcion1)

print('Valor analítico: ', valor_analitico)
print('Regla del trapecio: ', trapecio)
print('Trapecio compuesto (n=2): ', trapecio_comp1)
print('Trapecio compuesto (n=4): ', trapecio_comp2)
print('Regla de Simpson 1/3: ', simpson_13)
print('Regla de Simpson 3/8: ', simpson_38)
print('\n')

# ************************* PREGUNTA 2 ******************************
print('PREGUNTA 2: ')
print('\n')
x = np.array([1, 2, 3.25, 4.5, 6, 7, 8, 8.5, 9, 10])
y = np.array([5, 6, 5.5, 7, 8.5, 8, 6, 7, 7, 5])

distancia_recorrida = trapecio_no_equi(x,y)
print('La distancia recorrida es: ', distancia_recorrida)
print('\n')

# ************************* PREGUNTA 3 ******************************
print('PREGUNTA 3: ')
print('\n')
def funcion2(v):

    f = 0.2 + 25*v - 200*math.pow(v, 2) + 675*math.pow(v, 3) - 900*math.pow(v, 4) + 400*math.pow(v, 5)
    return f

a = 0
b = 0.8
n1 = 1 # 2 puntos, h1
n2 = 2 # 3 puntos, h2
n3 = 4 # 5 puntos, h3
trapecio_comp1 = trapecio_com_fn(a, b,  n1, funcion2)
trapecio_comp2 = trapecio_com_fn(a, b,  n2, funcion2)
trapecio_comp3 = trapecio_com_fn(a, b,  n3, funcion2)

print('Valor analítico:', 1.64053333)
print('Método de trapecio con n = 2: ', trapecio_comp1)
print('Método de trapecio con n = 3: ', trapecio_comp2)
print('Método de trapecio con n = 5: ', trapecio_comp3)

result_richardson = extrapolacion_richardson(a, b, n3, n2, funcion2)
print('Con extrapolacion de Richardson (h({})/h({}):'.format('n3', 'n2' ), result_richardson)
print('\n')
# *********************** INTEGRACION NUMERICA II **************************

print('*********************** INTEGRACION NUMERICA II *************************')

# ************************** PREGUNTA 1 ************************************
print('PREGUNTA 1: ')
print('\n')
def equation(x):

    f = 4/(1 + math.pow(x,2))
    return f

integracion_GL3 = gauss_legendre(0, 1, equation, 3)
integracion_GL4 = gauss_legendre(0, 1, equation, 4)
integracion_GL5 = gauss_legendre(0, 1, equation, 5)

integracion_GRL3 = gauss_radau(0, 1, equation, 3)
integracion_GRL4 = gauss_radau(0, 1, equation, 4)
integracion_GRL5 = gauss_radau(0, 1, equation, 5)

integracion_GLL3 = gauss_lobatto(0, 1, equation, 3)
integracion_GLL4 = gauss_lobatto(0, 1, equation, 4)
integracion_GLL5 = gauss_lobatto(0, 1, equation, 5)

print('Valor de pi: ', np.pi)
print('Integracion GL (3 puntos): ', integracion_GL3)
print('Integracion GL (4 puntos): ', integracion_GL4)
print('Integracion GL (5 puntos): ', integracion_GL5)
print('\n')
print('Integracion GRL (3 puntos): ', integracion_GRL3)
print('Integracion GRL (4 puntos): ', integracion_GRL4)
print('Integracion GRL (5 puntos): ', integracion_GRL5)
print('\n')
print('Integracion GLL (3 puntos): ', integracion_GLL3)
print('Integracion GLL (4 puntos): ', integracion_GLL4)
print('Integracion GLL (5 puntos): ', integracion_GLL5)
print('\n')

# ************************** PREGUNTA 2 ************************************
print('PREGUNTA 2: ')
print('\n')

t = np.array([200, 202, 204, 206, 208, 210])
thetha = np.array([0.75, 0.72, 0.70, 0.68, 0.67, 0.66])
r = np.array([5120, 5370, 5560, 5800, 6030, 6240])

step = 2

# ********************** primeras derivadas ******************************

rforward1 = met_forward(r, step, 1) 
rbackward1 = met_backward(r, step, 1) 
rcentrada1 = met_centrada(r, step, 1)

thetha_forward1 = met_forward(thetha, step, 1) 
thetha_backward1 = met_backward(thetha, step, 1) 
thetha_centrada1 = met_centrada(thetha, step, 1)

print('Primera derivada de r:')
print('\n')
print('forward: ', rforward1)
print('backward: ', rbackward1)
print('centrada: ', rcentrada1)
print('\n')
print('Primera derivada de thetha:')
print('\n')
print('forward: ', thetha_forward1)
print('backward: ', thetha_backward1)
print('centrada: ', thetha_centrada1)
print('\n')
# ********************* segundas derivadas *******************************

rforward2 = met_forward(r, step, 2)
rbackward2 = met_backward(r, step, 2) 
rcentrada2 = met_centrada(r, step, 2)

thetha_forward2 = met_forward(thetha, step, 2) 
thetha_backward2 = met_backward(thetha, step, 2) 
thetha_centrada2 = met_centrada(thetha, step, 2)

print('Segunda derivada de r:')
print('\n')
print('forward: ', rforward2)
print('backward: ', rbackward2)
print('centrada: ', rcentrada2)
print('\n')
print('segunda derivada de thetha:')
print('\n')
print('forward: ', thetha_forward2)
print('backward: ', thetha_backward2)
print('centrada: ', thetha_centrada2)
print('\n')
# ********************* vector v y a*****************************************

r_derivada1 = np.zeros(len(r))
r_derivada2 = np.zeros(len(r))
thetha_derivada1 = np.zeros(len(thetha))
thetha_derivada2 = np.zeros(len(thetha))

vc = np.array([]) # ESTE VECTOR SERVIRA PARA ASIGNAR UNA LETRA A CADA COMPONENTE, QUE INDIQUE CON QUE METODO FUE CALCULADO
# C: centrada
# F: forward
# B: backward

for i in range(len(r)):

    if rcentrada1[i] != 0:
        r_derivada1[i] = rcentrada1[i]
        vc = np.append(vc,'C')
    elif rforward1[i] != 0:
        r_derivada1[i] = rforward1[i]
        vc = np.append(vc,'F')
    else:   
        r_derivada1[i] = rbackward1[i]
        vc = np.append(vc,'B')
print('Vector primera derivada de r:')    
print('\n')
print(r_derivada1)
print(vc)
print('\n')

vc = np.array([])
for i in range(len(r)):

    if rcentrada2[i] != 0:
        r_derivada2[i] = rcentrada2[i]
        vc = np.append(vc,'C')
    elif rforward2[i] != 0:
        r_derivada2[i] = rforward2[i]
        vc = np.append(vc,'F')
    else:   
        r_derivada2[i] = rbackward2[i]
        vc = np.append(vc,'B')

print('Vector segunda derivada de r:')    
print('\n')
print(r_derivada2)
print(vc)
print('\n')

vc = np.array([])
for i in range(len(thetha)):

    if thetha_centrada1[i] != 0:
        thetha_derivada1[i] = thetha_centrada1[i]
        vc = np.append(vc,'C')
    elif thetha_forward1[i] != 0:
        thetha_derivada1[i] = thetha_forward1[i]
        vc = np.append(vc,'F')
    else:   
        thetha_derivada1[i] = thetha_backward1[i]
        vc = np.append(vc,'B')
    
print('Vector primera derivada de thetha:')    
print('\n')
print(thetha_derivada1)
print(vc)
print('\n')

vc = np.array([])
for i in range(len(thetha)):

    if thetha_centrada2[i] != 0:
        thetha_derivada2[i] = thetha_centrada2[i]
        vc = np.append(vc,'C')
    elif thetha_forward2[i] != 0:
        thetha_derivada2[i] = thetha_forward2[i]
        vc = np.append(vc,'F')
    else:   
        thetha_derivada2[i] = thetha_backward2[i]
        vc = np.append(vc,'B')
    
print('Vector segunda derivada de thetha:')    
print('\n')
print(thetha_derivada2)
print(vc)
print('\n')

print('Vectores componentes:')
print('\n')
v_er = np.zeros(len(r))
v_ethetha = np.zeros(len(thetha))
a_er = np.zeros(len(r))
a_ethetha = np.zeros(len(thetha))

for i in range(len(r)):
    v_er[i] = r_derivada1[i]
    v_ethetha[i] = r[i]*thetha_derivada1[i]
    a_er[i] = r_derivada2[i] - r[i]*math.pow(thetha_derivada1[i], 2)
    a_ethetha[i] = r[i]*thetha_derivada2[i] + 2*r_derivada1[i]*thetha_derivada1[i]

print('Componente radial de v: ', v_er)
print('Componente angular de v: ', v_ethetha)
print('Componente radial de a:', a_er)
print('Componente angular de a: ', a_ethetha)
