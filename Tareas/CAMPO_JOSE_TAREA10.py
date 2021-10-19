#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from CAMPO_MODULO_TAREA10 import *

#***************************  PREGUNTA 1  *****************************

a, b, T0, Tf, step, h, error = 0, 10, 40, 200, 1, 0.01, 0.00001
coef_vec = np.array([1, -1*(2+h*pow(step,2)), 1])

def f1(x):
    h = 0.01
    Ta = 20
    f = -h*Ta
    return f

T = cond_frontera_dirichlet(a, b, T0, Tf, step, coef_vec, f1, error)
print('\n')

#Gráfica
def Temp1(x):

    t = -53.4523*np.exp(-0.1*x) + 73.4523*np.exp(0.1*x) + 20
    return t

xdom = np.linspace(a,b,100)
temp = np.array([Temp1(x) for x in xdom])
n_pts = int((b-a)/step) + 1
x = np.linspace(a,b,n_pts)
T = np.insert(T, 0, T0)
T = np.insert(T, n_pts - 1, Tf)
plt.plot(xdom, temp, 'k-', x, T, 'bo-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función', 'Aproximación'), prop = {'size': 20}, loc='upper left')
plt.title('Condiciones de Dirichlet')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

# Errores por refinamiento:
# Profesor, los errores por refinamiento me salen algo extraños. De todas maneras le envio mi script para
# que lo pueda revisar. Estoy usando la norma infinita.

print('Error globlal:')
h_vec = np.array([0.25, 0.5, 1, 2])
error_global = np.zeros(len(h_vec))

for i in range(len(h_vec)):
    
    n_vec = int((b-a)/h_vec[i]) + 1
    coef_vec = np.array([1, -1*(2+h*pow(h_vec[i],2)), 1])
    T_vec = cond_frontera_dirichlet(a, b, T0, Tf, h_vec[i], coef_vec, f1, error)
    xdom_vec = np.linspace(a,b,n_vec) 
    T_analit = np.array([Temp1(x) for x in xdom_vec])
    T_analit = np.delete(T_analit, 0)
    T_analit = np.delete(T_analit, n_vec - 2)
    error_global[i] = max(T_vec - T_analit)

print(error_global)
plt.plot(h_vec, error_global, 'ko-')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()
#***************************  PREGUNTA 2  *****************************

a, b, alpha, Tf, step, h = 0, 10, 10, 200, 1, 0.01
coef_vec = np.array([1, -1*(2+h*pow(step,2)), 1])
    
def f2(x):
    h = 0.01
    Ta = 40
    f = -h*Ta
    return f

T = cond_frontera_newman(a, b, alpha, Tf, step, coef_vec, f2)
print('\n')

def Temp2(x):

    t = -36.2354*np.exp(-0.1*x) + 63.7646*np.exp(0.1*x) + 40
    return t

xdom = np.linspace(a,b,100)
temp = np.array([Temp2(x) for x in xdom])
n_pts = int((b-a)/step) + 1
x = np.linspace(a,b,n_pts)
T = np.insert(T, n_pts - 1, Tf)
plt.plot(xdom, temp, 'k-', x, T, 'ro-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función', 'Aproximación'), prop = {'size': 20}, loc='upper left')
plt.title('Condiciones de Newman')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

# Errores por refinamiento
# Profesor, los errores por refinamiento me salen algo extraños aquí también.

print('Error globlal:')
h_vec = np.array([0.25, 0.5, 1, 2])
error_global = np.zeros(len(h_vec))

for i in range(len(h_vec)):
    
    n_vec = int((b-a)/h_vec[i]) + 1
    coef_vec = np.array([1, -1*(2+h*pow(h_vec[i],2)), 1])
    T_vec = cond_frontera_newman(a, b, alpha, Tf, h_vec[i], coef_vec, f2)
    xdom_vec = np.linspace(a,b,n_vec) 
    T_analit = np.array([Temp1(x) for x in xdom_vec])
    T_analit = np.delete(T_analit, n_vec - 1)
    error_global[i] = abs(max(T_vec - T_analit))

print(error_global)
plt.plot(h_vec, error_global, 'ko-')
plt.yscale('log')
#plt.xscale('log')
plt.show()
#***************************  PREGUNTA 3  *****************************

def dT(x, f_vect):

    dT = f_vect[1]
    return dT

def dz(x, f_vect):

    h = 0.01 
    T_inf = 20
    dz = -1*h*(T_inf - f_vect[0])
    return dz

a, b, step = 0, 10, 2
T0, Tf = 40, 200
equ_vector = np.array([dT, dz])

#Supuesto 1
supuesto1 = 10
f0_vec = np.array([T0, supuesto1])
r_vec1 = s_ode_RK4(a, b, f0_vec, step, equ_vector)
n_pts1 = len(r_vec1)
T1 = np.array([r_vec1[i][0] for i in range(n_pts1)])

#Supuesto 2
supuesto2 = 20
f0_vec = np.array([T0, supuesto2])
r_vec2 = s_ode_RK4(a, b, f0_vec, step, equ_vector)
n_pts2 = len(r_vec2)
T2 = np.array([r_vec2[i][0] for i in range(n_pts2)])

#Calculamos el valor inicial de z
z =  supuesto1 + (supuesto2 - supuesto1)*(Tf - T1[n_pts1 - 1])/(T2[n_pts2-1] - T1[n_pts1-1])
f0_vec = np.array([T0, z])
r_vec3 = s_ode_RK4(a, b, f0_vec, step, equ_vector)
n_pts3 = len(r_vec3)
T3 = np.array([r_vec3[i][0] for i in range(n_pts3)])

tdom1 = np.linspace(a, b, n_pts1)
tdom2 = np.linspace(a, b, n_pts2)
tdom3 = np.linspace(a, b, n_pts3)
xdom = np.linspace(a,b,100)
temp = np.array([Temp1(x) for x in xdom])
plt.plot(tdom1, T1, 'b-', tdom2, T2, 'r-', tdom3, T3, 'g-*', xdom, temp, 'k-' )
plt.rc('axes', titlesize = 22)
plt.legend(('Supuesto1: {}'.format(supuesto1), 'Supuesto2: {}'.format(supuesto2), 'Aproximación', 'Función'), prop = {'size': 20}, loc='upper left')
plt.title('Método del disparo')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

#***************************  PREGUNTA 4  *****************************

B = np.array([[-4, 10], 
             [7,5]])
C = np.array([[1,2,-2], 
              [-2,5,-2], 
              [-6,6,-3]])

errorf = 0.00001
v1, eig_value1, error1, cont1 = metodo_potencias(B, errorf)
v2, eig_value2, error2, cont2 = metodo_potencias(C, errorf)

print('\n')
print('Matriz B: ', B)
print('\n')
print('Mayor vector propio')
print(v1)
print('Mayor valor propio')
print(eig_value1)
print('Número de iteraciones')
print(cont1)
print('\n')
print('Matriz C: ', C)
print('\n')
print('Mayor vector propio')
print(v2)
print('Mayor valor propio')
print(eig_value2)
print('Número de iteraciones')
print(cont2)













