#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from pylab import *
from CAMPO_MODULO_TAREA11 import *

#******************************* PREGUNTA 1 *************************************

def f(x,y):

    f=0
    return f

# Cambiando el número de puntos (npt), se puede refinar el cálculo

a, b, npt, alpha1, alpha2, errorf = 0, 100, 10, 0, 0, 0.0000001
x0, xf = 0, 4
y0, yf = 0, 4
Tx0 = np.linspace(a, b, npt)
Ty0 = np.linspace(a, b, npt)
x = np.linspace(x0,xf,npt)
y = np.linspace(y0,yf,npt)

T, Taux = equ_eliptica(x0, xf, npt, f, alpha1, alpha2, Tx0, Ty0, x, y, errorf)
Taux = np.reshape(Taux, (npt-1,npt-1))
Taux = Taux[::-1]

for i in range(1,npt):
    for j in range(1,npt):

        T[i][j-1] = Taux[i-1][j-1]

print('Campo de Temperaturas')        
print('\n')
print(T)

#Graficamos

data = T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.rc('axes', titlesize = 22)
plt.title('$MAPA\ DE\ CALOR\ (DIRICHLET\ -\ NEUMANN)$')
plt.xlabel('$Eje\ y$', size = 22)
plt.ylabel('$Eje\ x$', size = 22)
cax = ax.matshow(data, interpolation='nearest')
fig.colorbar(cax, label = '$TEMPERATURA (T)$')

plt.show()
