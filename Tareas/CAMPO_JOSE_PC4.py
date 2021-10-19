#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import gridspec
import math
from pylab import *
from CAMPO_MODULO_PC4 import *

#******************************* PREGUNTA 1 ************************************    

# PREGUNTA 1.B

def Fa(t):
    
    thau = 365
    f = 10 + 12*np.sin(2*np.pi*t/thau)
    return f

def Fb(t):

    f = 11
    return f

a, b, n_pts, step_time, n_time, alpha, errorf = 0, 20, 21, 5, 658, 0.1, 0.00001
T0 = np.zeros(n_pts)
T0[n_pts-1] = Fb(0)
for i in range(n_pts-1):
    T0[i] = 10


T, xdom =  crank_nicolson(a, b, Fa, Fb, n_pts, step_time, n_time, T0, alpha, errorf)

T = T[:,::-1] #Invertimos las columnas, pues se tomo de referenciac como punto inicial a A

# Graficamos para los distintos años

lista = () # Aquí se guardaran las etiquetas

for i in range(n_time):
    #Perfiles de temperatura
    if i%73 == 0:

        plt.plot(xdom, T[i], 'o-')
        lista = list(lista)
        lista.insert(i+1, 'Año {}'.format(i//73))
        lista = tuple(lista)

plt.rc('axes', titlesize = 22)
plt.title('$PREGUNTA\ 1.B:\ TEMPERATURA\ DE\ LA\ CORTEZA\ TERRESTRE\ vs\ PROFUNDIDAD$')
plt.xlabel('$x\ (m)$', size = 22)
plt.ylabel('$Temperatura\ (C°)$', size = 22)
plt.legend(lista, prop = {'size': 20}, loc='lower left')
plt.grid()
plt.show()   

# Graficamos para los años 3, 6, 9

plt.plot(xdom, T[3*73], xdom, T[6*73], xdom, T[9*73])
plt.title('$PREGUNTA\ 1.B:\ TEMPERATURA\ DE\ LA\ CORTEZA\ TERRESTRE\ vs\ PROFUNDIDAD$')
plt.xlabel('$x\ (m)$', size = 22)
plt.ylabel('$Temperatura\ (C°)$', size = 22)
plt.legend(('Año 3', 'Año 6', 'Año 9'), prop = {'size': 20}, loc='lower left')
plt.grid()
plt.show()

# PREGUNTA 1.C
print ('**************************** PREGUNTA 1 **************************')
print('\n')
print('1.C :')
print('\n')
print('Según lo observado en las graficas anteriores, podriamos concluir que la temperatura tiende a un estado')
print('estacionario. Sin embargo, nosotros hemos graficado perfiles según cada año. Si graficaramos, por ejemplo')
print('para cada mes de un año, como vemos a continuacion, veriamos que los perfiles, en realidad, varian de') 
print('forma periodica, por lo que no tendería a un estado estacionario. Podemos decir que este comportamiento') 
print('periodico se debe a la condicion de frontera variable (periodica) que tenemos en el extremo A.')

n_time = 74
T, xdom =  crank_nicolson(a, b, Fa, Fb, n_pts, step_time, n_time, T0, alpha, errorf)
T = T[:,::-1] #Invertimos las columnas, pues se tomo de referenciac como punto inicial a A

#Graficamos
lista = () # Aquí se guardaran las etiquetas

for i in range(n_time):
    #Perfiles de temperatura
    if i%6 == 0:

        plt.plot(xdom, T[i], 'o-')
        lista = list(lista)
        lista.insert(i+1, 'Mes {}'.format(i//6))
        lista = tuple(lista)

plt.rc('axes', titlesize = 22)
plt.title('$PREGUNTA\ 1.B:\ TEMPERATURA\ DE\ LA\ CORTEZA\ TERRESTRE\ vs\ PROFUNDIDAD$')
plt.xlabel('$x\ (m)$', size = 22)
plt.ylabel('$Temperatura\ (C°)$', size = 22)
plt.legend(lista, prop = {'size': 20}, loc='lower left')
plt.grid()
plt.show()   

#******************************* PREGUNTA 2 ************************************    

# Definimos las funciones 

def dens(x,y):

    d = 177.08*(10**(-12))*np.cos(3*np.pi*x)*np.sin(2*np.pi*y)
    return d

def Pot_E1(y, lamb1):

    Px0 = lamb1*pow(y, 2)
    return Px0

def Pot_E2(x, lamb2):

    Py0 = lamb2*pow(x, 3)
    return Py0

# Definimos las constantes y los dominios

e       = 8.854*(10**(-12)) # epsilon
lamb1   = 1
lamb2   = 1
Pot_E1y = 1
Pot_Ex1 = 1

x0, xf  = 0, 1
y0, yf  = 0, 1
xdom    = np.linspace(x0, xf, 5)
ydom    = np.linspace(y0, yf, 5)
n_ptsx  = len(xdom)
n_ptsy  = len(ydom)
step    = (xf-x0)/(5-1)


# Definimos el espacio S como una matriz de 5x5

Pot_E = np.zeros((n_ptsx, n_ptsy))

# Completamos la matriz con las condiciones de frontera

for j in range(n_ptsx):
    
    Pot_E[0][j] = Pot_Ex1

for j in range(n_ptsx):

    Pot_E[n_ptsy-1][j] = Pot_E2(xdom[j], lamb2)

for i in range(1, n_ptsy - 1):

    Pot_E[i][n_ptsx-1] = Pot_E1y

for i in range(1, n_ptsy - 1):

    Pot_E[i][0] = Pot_E1(ydom[n_ptsy-(1+i)], lamb1)

# Calculamos los potenciales dentro de la malla

A = np.array([[-4, 1, 0, 1, 0, 0, 0, 0, 0], \
              [ 1,-4, 1, 0, 1, 0, 0, 0, 0], \
              [ 0, 1,-4, 0, 0, 1, 0, 0, 0], \
              [ 1, 0, 0,-4, 1, 0, 1, 0, 0], \
              [ 0, 1, 0, 1,-4, 1, 0, 1, 0], \
              [ 0, 0, 1, 0, 1,-4, 0, 0, 1], \
              [ 0, 0, 0, 1, 0, 0,-4, 1, 0], \
              [ 0, 0, 0, 0, 1, 0, 1,-4, 1], \
              [ 0, 0, 0, 0, 0, 1, 0, 1,-4]])

A = 1/pow(step,2)*A
# Definimos una mtriz de densidad de puntos interiores
np_int = n_ptsx - 2 
Mdens = np.zeros((np_int, np_int))

for i in range(np_int):
    for j in range(np_int):

        Mdens[i][j] = dens(xdom[j+1], ydom[i+1])

# Invertimos la matriz Mdens, pues fue calculado en el orden inverso

Mdens = Mdens[::-1]
Mdens = -1*Mdens # multiplicamos por -1 para que sea más fácil trabajar

# Definimos el vector F

F = np.array([Mdens[2][0]/e - (Pot_E[4][1] + Pot_E[3][0])/pow(step,2), \
              Mdens[2][1]/e - (Pot_E[4][2])/pow(step,2)              , \
              Mdens[2][2]/e - (Pot_E[4][3] + Pot_E[3][4])/pow(step,2), \
              Mdens[1][0]/e - (Pot_E[2][0])/pow(step,2)              , \
              Mdens[1][1]/e                                          , \
              Mdens[1][2]/e - (Pot_E[2][4])/pow(step,2)              , \
              Mdens[0][0]/e - (Pot_E[1][0] + Pot_E[0][1])/pow(step,2), \
              Mdens[0][1]/e - (Pot_E[0][2])/pow(step,2)              , \
              Mdens[0][2]/e - (Pot_E[0][3] + Pot_E[1][4])/pow(step,2) ])

# Resolvemos el sistema
error = 0.00001
T, error, iteraciones = gauss_seidel(A, F, error) 

A_int = T.reshape(3,3)
A_int = A_int[::-1]

for i in range(np_int):
    for j in range(np_int):

        Pot_E[i+1][j+1] = A_int[i][j]

# Graficamos la Campo Potencial Eléctrico

Pot_E = Pot_E[::-1]

alpha = ['0', '0.25', '0.50', '0.75', '1.0'] #np.linspace(x0, xf, 5)

data = Pot_E

fig = plt.figure()
ax = fig.add_subplot(111)
plt.rc('axes', titlesize = 22)
plt.title('$PREGUNTA\ 2.B:\ POTENCIAL\ ELÉCTRICO\ -\ CONDICIONES\ DE\ DIRICHLET$')
plt.xlabel('$Eje\ y$', size = 22)
plt.ylabel('$Eje\ x$', size = 22)
cax = ax.matshow(data, interpolation='nearest')
fig.colorbar(cax, label = '$Potencial\ Eléctrico\ (V)$')
#fig.colorbar(label = 'Potencial Eléctrico')

#ax.set_xticklabels(['']+alpha)
#ax.set_yticklabels(['']+alpha)

plt.show()

#******************************* PREGUNTA 3 ************************************    

def dT(x, f_vect):
    
    dT = f_vect[1]
    return dT

def dz(x, f_vect):

    dz =  0.15*f_vect[0]
    return dz

def Temp(x):

    t = np.exp(-0.387298*x)*(236.983 + 3.01694*np.exp(0.774597*x))
    return t

a, b, step = 0, 10, 0.5
T0, Tf = 240, 150
equ_vector = np.array([dT, dz])

#Supuesto 1
supuesto1 = -80
f0_vec = np.array([T0, supuesto1])
r_vec1 = s_ode_RK4(a, b, f0_vec, step, equ_vector)
n_pts1 = len(r_vec1)
T1 = np.array([r_vec1[i][0] for i in range(n_pts1)])

#Supuesto 2
supuesto2 = -100
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
temp = np.array([Temp(x) for x in xdom])
plt.plot(tdom1, T1, 'b-', tdom2, T2, 'r-', tdom3, T3, 'g-*', xdom, temp, 'k-' )
plt.rc('axes', titlesize = 22)
plt.legend(('Supuesto1: {}'.format(supuesto1), 'Supuesto2: {}'.format(supuesto2), 'Aproximación', 'Función'), prop = {'size': 20}, loc='upper left')
plt.title('$PREGUNTA\ 3:\ METODO\ DEL\ DISPARO$')
plt.xlabel('$x\ (m)$', size = 22)
plt.ylabel('$Temperatura\ (C°)$', size = 22)
plt.grid()
plt.show()

#Perfiles de temperatura
fig, (ax0, ax1) = plt.subplots(nrows=2, gridspec_kw={'height_ratios':[2, 1],}, sharex=True)
ax0.grid()
ax0.plot(xdom, temp, 'k-', label = '$SOLUCIÓN\ ANALÍTICA$')
plt.title('Perfil de Temperatura')
ax0.legend( prop={'size': 15}, frameon = False )
ax1.axes.get_yaxis().set_visible(False)
ax1.imshow(np.atleast_2d(temp), cmap=plt.get_cmap('hot'), extent=(a, b, 0, 0.5))
plt.show()

fig, (ax0, ax1) = plt.subplots(nrows=2, gridspec_kw={'height_ratios':[2, 1],}, sharex=True)
ax0.grid()
ax0.plot(tdom3, T3, 'k*-', label = '$SOLUCIÓN\ APROX.$')
plt.title('Perfil de Temperatura')
ax0.legend( prop={'size': 15}, frameon = False )
ax1.axes.get_yaxis().set_visible(False)
ax1.imshow(np.atleast_2d(T3), cmap=plt.get_cmap('hot'), extent=(a, b, 0, 0.5))
plt.show()



    






