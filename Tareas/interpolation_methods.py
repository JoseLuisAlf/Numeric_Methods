
import matplotlib.pyplot as plt # from pylab import plot,show
import numpy as np
import math
from direct_methods import *

#********************** Monomios ****************************
def p_monomios(x, y):

    n = len(x)
    matrix_vander = np.ones((n, n))
    sol = np.zeros(n)

    for i in range(n):
        for j in range(1,n):
            matrix_vander[i][j] = math.pow(x[i], j)

    sol = eliminacion_gauss(matrix_vander, y)
    
    return sol
#************************************************************

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

#********************* Newton *******************************
def p_newton(x, y, v):

    def newton(x,i,v):
        new=1
        for j in range(i):
            new=new*(v-x[j])
        return new

    def newton_matriz(N,x):

        n =len(x)
        for i in range(n):
            for j in range(1,n):
                N[i][j]=newton(x,j,x[i])
    
    n = len(x)
    N=np.ones((n,n))
    newton_matriz(N, x)
    
    sol = sust_directa(N, y)

    f = sol[0]
    for i in range(1, n):
        f = f + sol[i]*newton(x,i,v)

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

#****************** Spline lineal ****************************
#**************** Spline cuadrática *************************
#******************* Spline Cúbica **************************

#************** Funciones para graficar *********************
def grafica_comparar(x1, y1, x2, y2):
    
    namexy = input('Nombre de la primera gráfica: ')
    namexdyd = input('Nombre de la segunda gráfica: ')
    Title = input('Título: ')
    xLabel = input('Eje x: ')
    yLabel = input('Eje y: ')
    #plt.yscale('log')
    plt.plot(x1, y1, 'k-', x2, y2, 'k:')
    plt.legend(loc='lower right')
    plt.legend((namexy,namexdyd), prop = {'size': 15}, loc='upper right')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(Title)
    plt.grid()
    plt.show()
#************************************************************
def graficar(x, y):

    namexy = input('Nombre de la gráfica: ')
    Title = input('Título: ')
    xLabel = input('Eje x: ')
    yLabel = input('Eje y: ')
    #plt.yscale('log')
    plt.plot(x, y, 'k:')
    plt.legend(loc='lower right')
    plt.legend((namexy), prop = {'size': 15}, loc='upper right')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(Title)
    plt.grid()
    plt.show()
#******************** Graficar Error ************************
def graficar_error(x, y1, y2):

    error = np.absolute(y1 -y2)

    bolean = input('Visualizar el error [s/n]: ')
    if bolean == 's':
        print(error)

    Title = input('Título: ')
    xLabel = input('Eje x: ')
    yLabel = input('Eje y: ')
    #plt.yscale('log')
    plt.plot(x, error, 'k:')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(Title)
    plt.grid()
    plt.show()






