#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from CAMPO_MODULO_PC3 import *

#************************** PREGUNTA 1 *******************************
'''
# a) Considerar las ecuaciones de Edward Lorenz para modelas la 
#    dinámica de fluido atmosféricos

# Definimos las derivadas de las funciones
def dx(t, f_vec):
    
    sigma = 10
    dx = -1*sigma*f_vec[0] + sigma*f_vec[1]
    return dx

def dy(t, f_vec):
    
    r = 28
    dy = r*f_vec[0] - f_vec[1] - f_vec[0]*f_vec[2]
    return dy

def dz(t, f_vec):
    
    b = 2.6667
    dz = -1*b*f_vec[2] + f_vec[0]*f_vec[1]
    return dz

# Definimos los parametros que pasaremos a la función
x0, xf, step = 0, 30, 0.00005
f0_vec = np.array([5,5,5])
equ_vector = np.array([dx, dy, dz])

# Llamamos a la función
r_vec = s_ode_RK4(x0, xf, f0_vec, step, equ_vector)
# Imprimimos el número de puntos, y los puntos generados
#print(r_vec)
print('PREGUNTA 1.a: ')
print('\n')
n_pts = len(r_vec)
print('Cantidad de puntos generados para cada función: ', n_pts)

# b) Grafique las soluciones x, y y z en función de t en una misma hoja
print('\n')
print('PREGUNTA 1.b: ')
print('\n')
print('Gráfica mostrada en pantalla: ')
# Graficamos
tdom = np.linspace(x0, xf, n_pts)
x = np.array([r_vec[i][0] for i in range(n_pts)])
#plt.plot(tdom, x, 'k-')
#plt.show()

y = np.array([r_vec[i][1] for i in range(n_pts)])
#plt.plot(tdom, y, 'k-')
#plt.show()

z = np.array([r_vec[i][2] for i in range(n_pts)])
#plt.plot(tdom, z, 'k-')
#plt.show()

subplots_adjust(hspace=0.000)
G = np.array([x, y, z])
label = np.array(['$x$', '$y$', '$z$'])
num_graficas = 3

for i in range(num_graficas):
    j = i+1
    grafica = subplot(num_graficas,1,j)
    plt.ylabel(label[i], size = 15)
    grafica.plot(tdom,G[i], label = '$Función$ ${}$'.format(j))
    plt.legend( prop = {'size': 15}, loc = 'upper right' )

plt.suptitle('$Graficas\ para\ el\ sistema\ de\ ecuaciones$')
plt.xlabel('$Tiempo$ $(t)$', size = 15)
plt.show()

# c) Grafique la proyección de la solución (x(t), y(t), z(t)) en el plano
#    xy y el plano xz

print('\n')
print('PREGUNTA 1.c: ')
print('\n')
print('Gráficas mostradas en pantalla: ')
# Proyección sobre el plano xy
plt.plot(x, y, 'b-')
plt.rc('axes', titlesize = 22)
plt.title('$Proyección$ $sobre$ $el$ $plano$ $xy$')
plt.xlabel('$x$', size = 22)
plt.ylabel('$y$', size = 22)
plt.grid()
plt.show()

# Proyección sobre el plano xz
plt.plot(x, z, 'r-')
plt.title('$Proyección$ $sobre$ $el$ $plano$ $xz$')
plt.xlabel('$x$', size = 22)
plt.ylabel('$z$', size = 22)
plt.grid()
plt.show()

#************************** PREGUNTA 2 *******************************

# b) Resolver el sistema para w1, w2, thetha1, thetha2

def dw1(t, f):
    
    g = 9.81
    l = 0.4
    dw1 = -1*( pow(f[2],2) * np.sin(2*f[0] - 2*f[1]) + 2*pow(f[3],2)*np.sin(f[0] - f[1]) + (g/l)*(np.sin(f[0] - 2*f[1]) + 3*np.sin(f[0])))/(3 - np.cos(2*f[0] - 2*f[1]))
    return dw1

def dw2(t, f):

    g = 9.81
    l = 0.4
    dw2 = (4*pow(f[2],2)*np.sin(f[0] - f[1]) + pow(f[3],2)*np.sin(2*f[0] - 2*f[1]) + 2*(g/l)*(np.sin(2*f[0] - f[1]) - np.sin(f[1])))/(3 - np.cos(2*f[0] - 2*f[1]))
    return dw2

def dthetha1(t, f):

    g = 9.81
    l = 0.4
    dthetha1 = f[2]
    return dthetha1

def dthetha2(t, f):

    g = 9.81
    l = 0.4
    dthetha2 = f[3]
    return dthetha2

# Definimos los parametros que pasaremos a la función
x0, xf, step = 0, 100, 0.005
f0_vec = np.array([np.pi/2,np.pi/2,0,0])
equ_vector = np.array([dthetha1, dthetha2, dw1, dw2])

# Llamamos a la función
r_vec = s_ode_RK4(x0, xf, f0_vec, step, equ_vector)
# Imprimimos el número de puntos, y los puntos generados
#print(r_vec)

print('\n')
print('PREGUNTA 2.b: ')
print('\n')
n_pts = len(r_vec)
print('La cantidad de puntos es, para cada función: ', n_pts)

# Graficamos
tdom = np.linspace(x0, xf, n_pts)
thetha1 = np.array([r_vec[i][0] for i in range(n_pts)])
thetha2 = np.array([r_vec[i][1] for i in range(n_pts)])
w1 = np.array([r_vec[i][2] for i in range(n_pts)])
w2 = np.array([r_vec[i][3] for i in range(n_pts)])

# c) Evolución temporal de T, V y E

print('\n')
print('PREGUNTA 2.c: ')
print('\n')
print('Gráfica mostrada en pantalla: ')

def T(thetha1, thetha2, w1, w2):
    
    m = 1
    l = 0.4
    g = 9.81
    T = 0.5*m*pow(l, 2)*(2*pow(w1,2) + pow(w2,2) + 2*w1*w2*np.cos(thetha1 - thetha2))
    return T

def V(thetha1, thetha2, w1, w2):
    
    m = 1
    l = 0.4
    g = 9.81
    V = -m*g*l*(2*np.cos(thetha1) + np.cos(thetha2))
    return V

def E(T,V):

    E = T + V
    return E

eT = np.zeros(n_pts)
for i in range(n_pts):  
    
    eT[i] = T(thetha1[i], thetha2[i], w1[i], w2[i])

eV = np.zeros(n_pts)
for i in range(n_pts):  
    
    eV[i] = V(thetha1[i], thetha2[i], w1[i], w2[i])

eE = np.zeros(n_pts)
for i in range(n_pts):

    eE[i] = E(eT[i], eV[i])

plt.plot(tdom, eT, 'b-', tdom, eV, 'r-', tdom, eE, 'g-')
plt.rc('axes', titlesize = 22)
plt.legend(('$T$', '$V$', '$E$'), prop = {'size':20}, loc="center right", borderaxespad= -5.5)
plt.title('$Energía$ $($ $T$, $V$, $E$ $)$ $vs$ $t$')
plt.ylabel('$Joules$ $(J)$', size = 22)
plt.xlabel('$Tiempo$ $(t)$', size = 22)
plt.grid()
plt.show()

# Fundamento de lo obtenido

print('\n')
print('Análisis: ')
print('\n')
print('De la grafica obtenida, podemos ver que la energía se conserva. En todo momento la E marca una energía')
print('total igual a cero, esto debido a que las condiciones iniciales para los angulos thetha1 y thetha2')
print('son iguales a pi/2. Quiere decir que ambas masas empiezan sobre el eje x y con velocidades iguales a cero.')

#************************** PREGUNTA 3 *******************************

# a) Calcule la capacidad calorifica Cv para T = 30 K usando el método del 
#    trapecio y Simpson compuestos, y con 100 intervalos en ambos casos.

def Cv(T, V, dens, I):

    kb = 1.3806*(10**(-23)) 
    Tdebye = 428
    Cv = 9*V*dens*kb*pow(T/Tdebye, 3)*I
    return Cv

# Definiendo parametros
Tdebye = 428
T = 30
V = 0.0001 # 100 cm^3
dens = 6.22*(10**28)
x0 = 0.0000001
xf = Tdebye/T
n_int = 99
'''
# Definiendo la función a integrar y uso de los métodos señalados
def f1(x):

    epsilon = 8.85*10**(-12)
    elect = 1.6*10**(-19)
    kboltz = 1.38*10**(-23)
    Temp = 70000000
    Aconst = elect/(4*np.pi*8.85*10**(-12))
    n_0 = 10**(20)
    const = elect*Aconst/(kboltz*Temp)
    debye = (kboltz*epsilon*70000000/(10**(20)*elect**2))**(0.5) #debye
    f = 4*np.pi*(x**2)*n_0*np.exp((const/x)*np.exp(-x/debye))
    #f = pow(x,4)*np.exp(x)/pow(np.exp(x) - 1, 2)
    return f

def f2(x):

    epsilon = 8.85*10**(-12)
    elect = 1.6*10**(-19)
    kboltz = 1.38*10**(-23)
    Temp = 10000000
    Aconst = elect/(4*np.pi*8.85*10**(-12))
    n_0 = 10**(20)
    const = elect*Aconst/(kboltz*Temp)
    debye = (kboltz*epsilon*10000000/(10**(20)*elect**2))**(0.5) #debye
    f = 4*np.pi*(x**2)*n_0*np.exp((const/x)*np.exp(-x/debye))
    #f = pow(x,4)*np.exp(x)/pow(np.exp(x) - 1, 2)
    return f

def f3(x):

    epsilon = 8.85*10**(-12)
    elect = 1.6*10**(-19)
    kboltz = 1.38*10**(-23)
    Temp = 1000000
    Aconst = elect/(4*np.pi*8.85*10**(-12))
    n_0 = 10**(20)
    const = elect*Aconst/(kboltz*Temp)
    debye = (kboltz*epsilon*1000000/(10**(20)*elect**2))**(0.5) #debye
    f = 4*np.pi*(x**2)*n_0*np.exp((const/x)*np.exp(-x/debye))
    #f = pow(x,4)*np.exp(x)/pow(np.exp(x) - 1, 2)
    return f

def numberp(debye):

    n0 = 10**(20)
    f = 4/3*np.pi*debye**3*n0
    return f

elect = 1.6*10**(-19)
kboltz = 1.38*10**(-23)
epsilon = 8.85*10**(-12)
n_int=99
x0 = 0.00000000001
xf1 = (kboltz*epsilon*70000000/(10**(20)*elect**2))**(0.5) #debye
xf2 = (kboltz*epsilon*10000000/(10**(20)*elect**2))**(0.5) #debye
xf3 = (kboltz*epsilon*1000000/(10**(20)*elect**2))**(0.5) #debye
#I1 =  trapecio_com_fn(x0, xf, n_int, f) 
I1 =  met_simpson38_comp(x0, xf1, n_int, f1) 
I2 =  met_simpson38_comp(x0, xf2, n_int, f2) 
I3 =  met_simpson38_comp(x0, xf3, n_int, f3) 
#Cv1 = Cv(T, V, dens, I1)
#Cv2 = Cv(T, V, dens, I2)
print(xf1,I1)
print(xf2,I2)
print(xf3,I3)
xdom = np.array([xf1,xf2,xf3])
xdom1 = np.linspace(0.00000000001, 0.001, 20)
ydom = np.array([I1,I2,I3])
ydom1 = np.array([numberp(debye) for debye in xdom])
ydom2 = np.array([f1(i) for i in xdom1])
print(ydom1)
plt.plot(xdom,ydom, 'ro', xdom, ydom1, 'g*')
plt.show()
plt.plot(xdom1,ydom2,'r-')
plt.show()
'''
print('\n')
print('PREGUNTA 3.a: ')
print('\n')
print('Debido a que, en el límite inferior (T0 = 0) la funcion integrando tiende a infinito, ')
print('usaremos un punto muy cercano pero no igual a cero -> T0 = {}.'.format(x0))
print('\n')
print('Cv (Trapecio compuesto)    = ', Cv1)
print('Cv (Simpson 3/8 compuesto) = ', Cv2)

# b) Muestre la gráfica Cv vs T para un rango de T entre 5k y  500k usando
#    100 puntos.
print('\n')
print('PREGUNTA 3.b: ')
print('\n')
print('Grafico mostrado por pantalla: ')
T0 = 5
Tf = 500
n_pts = 100
n_int = n_pts - 1
Tdom = np.linspace(T0, Tf, n_pts)

Cv_vec1 = np.zeros(n_pts)

for i in range(n_pts):

    xf = Tdebye/Tdom[i]
    I = met_simpson38_comp(x0, xf, n_int, f)
    Cv_vec1[i] = Cv(Tdom[i], V, dens, I)

Cv_vec2 = np.zeros(n_pts)

for i in range(n_pts):

    xf = Tdebye/Tdom[i]
    I = trapecio_com_fn(x0, xf, n_int, f)
    Cv_vec2[i] = Cv(Tdom[i], V, dens, I)

plt.plot(Tdom, Cv_vec1, 'k-', Tdom, Cv_vec2, 'b-')
plt.rc('axes', titlesize = 22)
plt.legend(('$Simpson\ 3/8\ comp.$', 'Trapecio comp.'), prop = {'size':20}, loc="center right")
plt.title('$Capacidad\ caloŕifica\ (Cv)\ vs\ Temperatura\ (T)$')
plt.xlabel('$Temperatura$ $(T)$', size = 22)
plt.ylabel('$Cv$ $(J/K)$', size = 22)
plt.grid()
plt.show()

# c) A partir de los datos obtenidos para Cv en el rango de temperaturas
#    [5k; 500k], estime el valor de Cv cuando T = 0k

print('\n')
print('PREGUNTA 3.c: ')
print('\n')
print('Usamos puntos de Chebyshev para la interpolacion con polinomios de Lagrange')
print('y usamos el método de Simpson 3/8 compuesto para la integral.')
print('\n')
print('Grafico mostrado por pantalla: ')

# Usamos puntos de Chevyshev para interpolar la curva

Tdom_cheby = puntos_chebyshev(T0, Tf, n_pts + 1)
Cv_cheby = np.zeros(n_pts + 1)

for i in range(n_pts + 1):

    xf = Tdebye/Tdom_cheby[i]
    I = met_simpson38_comp(x0, xf, n_int, f)
    Cv_cheby[i] = Cv(Tdom_cheby[i], V, dens, I)

T0_aux = 0.0000001 # T donde queremos extrapolar
Tdom_aux = np.linspace(T0_aux, Tf, n_pts)
Cv_aux = np.array([p_lagrange(Tdom_cheby, Cv_cheby, v) for v in Tdom_aux])

T = Tdom_aux[0]
Cv = Cv_aux[0]


plt.plot(Tdom_aux, Cv_aux, 'k-', label = '$Lagrange\ (P.\ Chebyshev)$')
plt.rc('axes', titlesize = 22)
plt.legend(prop = {'size':20}, loc="lower right")
plt.title('$Extrapolacion\ de\ la\ Cv\ vs\ T$')
plt.xlabel('$Temperatura$ $(T)$', size = 22)
plt.ylabel('$Cv$ $(J/K)$', size = 22)
plt.text(300, 100, '$T({})\ =\ {}$'.format(T, Cv), fontsize = 20 ) 
plt.grid()
plt.show()

'''
