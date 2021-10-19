#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from CAMPO_MODULO_TAREA9 import *

# *********************** PREGUNTA 1 ************************************
x0, y0, xf, step, errorf = 0, 1, 2, 0.1, 0.00001

def dev_equation(x, y):

    f = y*math.pow(x, 3) - 1.5*y
    return f

def func_equation(x):
    
    f = math.exp(math.pow(x, 4)/4 - 1.5*x)
    return f

u1 = np.linspace(x0, xf, 100)
v1 = np.array([func_equation(x) for x in u1])

u2, v2 = euler_method(x0, y0, xf, step, dev_equation)
u3, v3, iteraciones = heun_method(x0, y0, xf, step, dev_equation, errorf)
u4, v4 = runge_kutta(x0, y0, xf, step, dev_equation, 2, 2/3)
u5, v5 = runge_kutta(x0, y0, xf, step, dev_equation, 4, 0)

plt.plot(u1, v1, 'k-', u2, v2, 'bs-', u3, v3, 'go-', u4, v4, 'r^-', u5, v5, 'cp-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función', 'Euler', 'Heun', 'RK2 (2/3)', 'RK4'), prop = {'size': 20}, loc='upper left')
plt.title('Métodos de resolución para EDOs')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

#************************ PREGUNTA 2 ************************************

def df1(x, y_vec):
    f = 999*y_vec[0] + 1999*y_vec[1]
    return f

def df2(x,y_vec):
    f = -1000*y_vec[0] - 2000*y_vec[1]
    return f

def func1(x):
    f = math.exp(-1000*x)*(3998*math.exp(999*x) - 2999)/999
    return f

def func2(x):
    f = math.exp(-1000*x)*(2999 - 2000*math.exp(999*x))/999
    return f
# Euler explícito

x0, xf, step1 = 0, 0.2, 0.0005
y0_vec = np.array([1, 1])
equ_vector = np.array([df1, df2])

r1 = s_ode_eulerexp(x0, xf, y0_vec, step1, equ_vector)


#Euler implícito
step2 = 0.005
coef1 = np.array([999, 1999])
coef2 = np.array([-1000, -2000])

r2 = s_ode_eulerimp(x0, xf, y0_vec, step2, coef1, coef2)

# Gráficas
xdom = np.linspace(x0, xf, 1000)
ydom1 = np.array([func1(x) for x in xdom])
ydom2 = np.array([func2(x) for x in xdom])

n_pt1 = int((xf-x0)/step1 + 1)
n_pt2 = int((xf-x0)/step2 + 1)
xdom_e1 = np.linspace(x0, xf, n_pt1)
xdom_e2 = np.linspace(x0, xf, n_pt2)
ydom1_ee = np.array([r1[v][0] for v in range(n_pt1)]) 
ydom2_ee = np.array([r1[v][1] for v in range(n_pt1)]) 
ydom1_ei = np.array([r2[v][0] for v in range(n_pt2)]) 
ydom2_ei = np.array([r2[v][1] for v in range(n_pt2)]) 

plt.plot(xdom, ydom1, 'b-')
plt.rc('axes', titlesize = 22)
plt.title('Función1 (X1)')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

plt.plot(xdom, ydom2, 'r-')
plt.rc('axes', titlesize = 22)
plt.title('Función2 (X2)')
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()
#Euler explícito
plt.plot(xdom, ydom1, 'b-', xdom_e1, ydom1_ee, 'ko-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función1', 'Euler explícito'), prop = {'size': 20}, loc='lower left')
plt.title('Euler explícito (h = {})'.format(step1))
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

plt.plot(xdom, ydom2, 'r-', xdom_e1, ydom2_ee, 'ko-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función2', 'Euler explícito'), prop = {'size': 20}, loc='upper left')
plt.title('Euler explícito (h = {})'.format(step1))
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

# Euler implícito
plt.plot(xdom, ydom1, 'b-', xdom_e2, ydom1_ei, 'ko-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función1', 'Euler implícito'), prop = {'size': 20}, loc='lower right')
plt.title('Euler implícito (h = {})'.format(step2))
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

plt.plot(xdom, ydom2, 'r-', xdom_e2, ydom2_ei, 'ko-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función2', 'Euler implícito'), prop = {'size': 20}, loc='upper right')
plt.title('Euler implícito (h = {})'.format(step2))
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()
#************************ PREGUNTA 3 ************************************

def dfunc(x,y):

    f = -0.5*y + math.exp(-x)
    return f
def func(x):
    f = 12*math.exp(-0.5*x) - 2*math.exp(-x)
    return f

x0, xf, y0, yb, step, errorf = 2, 3, 4.143883, 5.222138, 0.5, 0.01
x, y, iteraciones1 = heun_modify(x0, xf, y0, yb, step, dfunc, errorf)
m, n, iteraciones2 = heun_method(x0, y0, xf, step, dfunc, errorf)

xdom = np.linspace(x0, xf, 1000)
ydom = np.array([func(x) for x in xdom])
x = np.delete(x,0)
y = np.delete(y,0)

plt.plot(xdom, ydom, 'k-', x, y, 'bo-', m, n, 'ro-')
plt.rc('axes', titlesize = 22)
plt.legend(('Función', 'Heun modificado', 'Heun sin mod.'), prop = {'size': 20}, loc='upper right')
plt.title('Heun modificado (h = {}, error = {})'.format(step, errorf))
plt.xlabel('Eje x', size = 22)
plt.ylabel('Eje y', size = 22)
plt.grid()
plt.show()

