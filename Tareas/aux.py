
#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from differentialeq_methods import *
# prueba 1

def dequation(x, y):

    f = -2*x**3 + 12*x**2 - 20*x + 8.5 + 0*y
    return f

def equation(x, y):
    
    f = -0.5*x**4 + 4*x**3 - 10*x**2 + 8.5*x + 1 + 0*y
    return f

def dequation1(x, y):

    f = 4*math.exp(0.8*x) - 0.5*y
    return f

def equation1(x, y):
    
    f = (4/1.3)*(math.exp(0.8*x) - math.exp(-0.5*x)) + 2*math.exp(-0.5*x)
    return f
'''
x0, y0, xf, step = 0, 1, 4, 0.5

x, y = euler_method(x0, y0, xf, step, dequation)
print(x)
print(y)
print('\n')

xdom = np.linspace(0, 4, 100)
ydom = np.array([equation(x, 1) for x in xdom])
'''

x0, y0, xf, step, errorf = 0, 2, 4, 1, 0.00001

x1, y1, iteraciones = heun_method(x0, y0, xf, step, dequation1, errorf)
print(x1)
print(y1)
print(iteraciones)
print('\n')
'''
x2, y2 = midpoint_method(x0, y0, xf, step, dequation)
print(x2)
print(y2)

plt.plot(xdom, ydom, 'k-', x, y, 'ko-', x1, y1, 'co-', x2, y2, 'ko-')
plt.show()

#************************************************************************

x0, y0, xf, step = 0, 1, 4, 0.5

a1, b1 = runge_kutta(x0, y0, xf, step, dequation, 2, 1/2) # heun simple?
a2, b2 = runge_kutta(x0, y0, xf, step, dequation, 2, 1)
a3, b3 = runge_kutta(x0, y0, xf, step, dequation, 2, 2/3)
a4, b4 = runge_kutta(x0, y0, xf, step, dequation, 3, 0) 
a5, b5 = runge_kutta(x0, y0, xf, step, dequation, 4, 0) 
a6, b6 = runge_kutta(x0, y0, xf, step, dequation, 5, 0) 

plt.plot(xdom, ydom, 'k-', x, y, 'ko-', a1, b1, 'cs-', a2, b2, 'bo-', a3, b3, 'b^-')
plt.show()
print('\n')
print(a6)
print(b6)
plt.plot(xdom, ydom, 'k-', a4, b4, 'bo-', a5, b5, 'bs-', a6, b6, 'r^-')
plt.show()

#Ejemplo: función rígida

def dfunc(x, y):

    f = -1000*y + 3000 -2000*math.exp(-x)
    return f

def func(x):

    f = 3 - 0.998*math.exp(-1000*x) - 2.002*math.exp(-x)
    return f

xdom = np.linspace(0,4,100)
ydom = np.array([func(v) for v in xdom])

x0, y0, xf, step = 0, 0, 4, 0.0005 # El paso es fundamental para controlar la estabilidad
x, y = euler_method(x0, y0, xf, step, dfunc)

plt.plot(xdom, ydom, 'k-', x, y, 'bo-')
plt.show()
'''
#Ejemplo2:
n = 2 # numero de equaciones

def df1(x, y_vec):
    f = -0.5*y_vec[0]
    return f

def df2(x,y_vec):
    f = 4 - 0.3*y_vec[1] - 0.1*y_vec[0]
    return f
x0, xf, step = 0, 2, 0.5
y0_vec = np.array([4, 6])
equ_vector = np.array([df1, df2])

r1 = s_ode_RK4(x0, xf, y0_vec, step, equ_vector)
r2 = s_ode_eulerexp(x0, xf, y0_vec, step, equ_vector)
print(r1)
print(r2)

x0, xf, step = 0, 2, 0.05
y0_vec = np.array([52.29, 83.82])
coef1 = np.array([-5, 3])
coef2 = np.array([100, -301])

r3 = s_ode_eulerimp(x0, xf, y0_vec, step, coef1, coef2)
print(r3)

def func1(x):
    f = 52.96*math.exp(-3.9899*x) - 0.67*math.exp(-302.0101*x)
    return f
def func2(x):
    f = 17.83*math.exp(-3.9899*x) + 65.99*math.exp(-302.0101*x)
    return f

xdom = np.linspace(x0, xf, 100)
ydom1 = np.array([func1(x) for x in xdom])
ydom2 = np.array([func2(x) for x in xdom])
n_pt = int((xf-x0)/step +1)
xdom_e = np.linspace(x0, xf, n_pt)
ydom1_e = np.array([r3[v][0] for v in range(n_pt)]) 
print(ydom1_e)
ydom2_e = np.array([r3[v][1] for v in range(n_pt)]) 
print(ydom2_e)
plt.plot(xdom, ydom1, 'b-', xdom_e, ydom1_e, 'ko-')
plt.show()

plt.plot(xdom, ydom2, 'r-', xdom_e, ydom2_e, 'ko-')
plt.show()

#Pregunta3  
def equat(x,y):
    f = 4*math.exp(0.8*x) - 0.5*y
    return f

x0, xf, y0, yb, step, errorf = 0, 4, 2, -0.3929953, 1, 0.00001
x, y, iteraciones = heun_modify(x0, xf, y0, yb, step, equat, errorf)
print(x)
print(y)
print(iteraciones)
