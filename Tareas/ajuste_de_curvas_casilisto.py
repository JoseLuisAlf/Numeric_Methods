
from numpy import *
import matplotlib.pyplot as plt # from pylab import plot,show
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
#/////////////////////////////   FUNCIONES A USAR   //////////////////////////////////

#************************************ Módulo: Regresión lineal ***********************
def XYdot_sum(x ,y):

    result = 0
    for i in range(len(x)):
        result = result + x[i] * y[i]
    return result

def XXdot_sum(x):
    result = 0
    for i in range(len(x)):
        result = result + math.pow(x[i], 2)
    return result

def vector_sum(x):
    result = 0
    for i in range(len(x)):
        result = result + x[i]
    return result

def vector_average(x):
    
    result = vector_sum(x) / len(x)
    return result
        
def linear_regression(x, y):

    coef0, coef1 = 0, 0
    coef1 = ( len(x) * XYdot_sum(x, y) - vector_sum(x) * vector_sum(y) )/ \
         ( len(x) * XXdot_sum(x) - math.pow(vector_sum(x), 2) )
    coef0 = vector_average(y) - coef1 * vector_average(x)
    return coef0, coef1
#********************** Regresión lineal - Cuantificación del error *************************

def Sr(x, y, a0, a1):

    result = 0
    for i in range(len(x)):
        result = result + math.pow((y[i] - a0 - a1 * x[i]), 2)
    return result

def SDeviation_ajuste(Sr, x):

    S_yx = math.sqrt(Sr / (len(x) - 2))
    return S_xy

def St(y):

    result = 0
    for i in range(len(y)):
        result = result + math.pow((y[i] - vector_average(y)), 2)
    return result

def SDeviation_statistics(y):

    Sy = math.sqrt(St(y) / (len(y) - 1))
    return Sy

def determination_coefficient(x, y, a0, a1):

    r = math.sqrt((St(y) - Sr(x, y, a0, a1)) / St(y))
    return r

#**************************** Módulo: Regresión no lineal **************************

#********************************** Funciones a usar *******************************
def parameters_deriv(x, p1, p2, i):

    if i == 1:
        
        # f = 1 - math.exp(-p2*x)
        f = x * math.exp(p2 * x)
        return f

    elif i == 2:
    
        #f = p1*x*math.exp(-p2*x)
        f= p1 * math.pow(x, 2) * math.exp(p2 * x)
        return f

    else:
        pass

#**********************
def Transposed_matrix(X, m, n):

    new_X = np. zeros((n, m))

    for i in range(n):
        for j in range(m):
            new_X[i][j] = X[j][i]
    return new_X
#***********************
def Dot_matrix(A, B, m, aux, n):

    C = np.zeros((m, n))

    for i in range(m):
        for j in range(n):

            for k in range(aux):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]

    return C
#**************************************
def eliminacion_gauss(M,const,n):
    M1=np.zeros((n,n+1))
    for i in range(n):
        M1[i]=np.concatenate((M[i],[const[i]]))
    #print(M1)
    #print('\n')

    for i in range(0,n-1):
        for j in range(1,n-i):
            aux=[M1[j+i][i]/M1[i][i]*s for s in M1[i]]
            #print(aux)
            for l in range(n+1):
                M1[j+i][l]=M1[j+i][l]-aux[l]
    return M1
#*****************************************************
def sust_inversa(M,const,sol):
    for i in range(len(M)):
        sol[len(M)-1-i]=const[len(M)-i-1]/M[len(M)-i-1][len(M)-i-1]
        for j in range(len(M)-i,len(M)):
            sol[len(M)-1-i]=sol[len(M)-1-i]-M[len(M)-1-i][j]*sol[j]/M[len(M)-1-i][len(M)-1-i]
#*********************************************************
def Sr_new(x, y, a0, a1):

    result = 0
    for i in range(len(x)):
        result = result + math.pow((y[i] - model_function(x[i], a0, a1)), 2)
    return result
#************************************************************
def nonlinear_regression(x, y, num_parameters, initial_value, iterations):

    num_data = len(x)
    sol = np.zeros(num_parameters)
    new_parat = initial_value
    Sr_values = np.array([Sr_new(x, y, new_parat[0], new_parat[1])])
    cont = 1
    condition = True
    while condition:
        p1 = new_parat[0] # alpha4, Parametro 1
        p2 = new_parat[1] # betha4, Parametro 2

        Z = np.zeros((num_data, num_parameters))

        for i in range(num_data):
            for j in range(num_parameters):
                Z[i][j] = parameters_deriv(x[i], p1, p2, j+1)

        #print(Z)
        #print('\n')


        Z_transposed = Transposed_matrix(Z, num_data, num_parameters)
        #print(Z_transposed)
        #print('\n')

        ZT_Z = Dot_matrix(Z_transposed, Z, num_parameters, num_data, num_parameters)
        #print(ZT_Z)
        #print('\n')

#****************** Matriz D *********************
        D = np.zeros(len(y))
        D_model = np.zeros(len(y))

        for i in range(len(y)):
            D_model[i] = model_function(x[i], p1, p2)

        #print(D_model)
        #print('\n')

        for i in range(len(y)):
            D[i] = y[i] - D_model[i]

        #print(D)
        #print('\n')

        ZT_D = np.dot(Z_transposed, D)
        #print(ZT_D)
        #print('\n')

        M = eliminacion_gauss(ZT_Z, ZT_D, num_parameters)
        #print(M)
        #print('\n')

        new_D = np.zeros(num_parameters)
        for i in range(num_parameters):
            new_D[i] = M[i][num_parameters]
        #print(new_D)
        #print('\n')

        M = np.delete(M, num_parameters, axis = 1)
        #print(M)

        sust_inversa(M, new_D, sol)

        #print(sol)

        new_parat = new_parat + sol
        #print('\n')
        Sr_values = np.append(Sr_values, Sr_new(x, y, new_parat[0], new_parat[1]))    
    
        if cont == iterations:
            condition = False
        else:
            cont += 1
        #if Sr_new(x, y) > 0.0001:
            #condition = False
    return Sr_values, new_parat
#///////////////////////////////////    PROBLEMAS    //////////////////////////////
'''
#************************** PROBLEMA 1 **********************************
x = np.array([0.5, 1, 2, 3, 4])
y = np.array([10.4, 5.8, 3.3, 2.4, 2])
print('PROBLEMA 1\n')
print('x = ', x)
print('y = ', y) 
print('\n')
print('Ecuación del modelo propuesto:\n ')
print('          y = [ ( a + x^(0.5) ) / b * x^(0.5) ]^2\n')
print('Linealizando, tenemos la siguiente ecuacion de la recta:\n')
print('          y\' = a0 + a1*x\'\n')
print('donde:\n')
print('          y\' = y^(0.5)')
print('          a0 = 1/b')
print('          a1 = a/b')
print('          x\' = 1/x^(0.5)\n')
print('Aplicamos regresión lineal con los nuevos valores:\n')

x1 = np.copy(x)**(-0.5)
y1 = np.copy(y)**(0.5)

print('x\' = ', x1)
print('y\' = ', y1)
print('\n')
print('Tenemos que:\n')

a0, a1 = linear_regression(x1, y1)

print('          a0 = ', a0)
print('          a1 = ', a1)
print('\n')
#**************************************
def lineal_function(x, a0, a1):
    
    f = a0 + a1*x
    return f
#**************************************
plt.grid()
plt.plot(x1, y1, 'bo', x1, [lineal_function(i, a0, a1) for i in x1], 'k-')
plt.show()

print('Luego:\n')
b = 1 / a0
a = a1 / a0

print('          a = a0/a1 = ', a)
print('          b = 1/a0 = ', b)
print('\n')
print('La ecuación del modelo propuesto sería:\n ')
print('          y = [ ( {} + x^(0.5) ) / {} * x^(0.5) ]^2\n'.format(a, b))
#**************************************
def model_function_P1(x, a, b):

    f =  math.pow((a + math.pow(x, 0.5))/(b * math.pow(x, 0.5)), 2)
    return f
#**************************************
plt.grid()
plt.plot(x, y, 'bo', x, [model_function_P1(i, a, b) for i in x], 'k-')
plt.show()

print('Coeficiente de determinación:\n')
print('          r = ', determination_coefficient(x1, y1, a0, a1))
print('\n')

#************************* PROBLEMA 2 *************************************

#********************* Regresion lineal ***********************************

x = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
y = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])
y2 = np.zeros(len(x))
for i in range(len(x)):
    y2[i] = math.log(y[i]/x[i])

a0, a1 = linear_regression(x, y2)

#**************************************
def lineal_function(x, a0, a1):
    
    f = a0 + a1*x
    return f
#**************************************
plt.grid()
plt.plot(x, y2, 'bo', x, [lineal_function(i, a0, a1) for i in x], 'k-')
plt.show()

alpha4 = math.exp(a0)
betha4 = a1

#***************************
def model_function(x, p1, p2):

    #f = p1*(1 - math.exp(-p2*x))
    f = p1 * x * math.exp(p2 * x)
    return f
#**************************************
xdom = np.arange(0, 2, 0.01)
plt.grid()
plt.plot(x, y, 'bo', xdom, [model_function(i, alpha4, betha4) for i in xdom], 'k-')
plt.show()

print('Coeficiente de determinación:\n')
print('          r = ', determination_coefficient(x, y2, a0, a1))
print('\n')
print(Sr_new(x, y, a0, a1))
print('\n')
print(alpha4)
print(betha4)
print('\n')
#********************** Regresion no lineal ***********************************
num_parameters = 2
iterations = 15
initial_value = np.array([1,1])
parameters = np.zeros(num_parameters)
Sr_values = np.zeros(iterations)

Sr_values, parameters = nonlinear_regression(x, y, num_parameters, initial_value, iterations)
print(Sr_values)
print(parameters)
#**************************************
plt.grid()
plt.plot(x, y, 'bo', xdom, [model_function(i, alpha4, betha4) for i in xdom], 'k-', xdom, [model_function(j, parameters[0], parameters[1]) for j in xdom], 'k:')
plt.show()
#*****************************************************************************
# x = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
# y = np.array([0.28, 0.57, 0.68, 0.74, 0.79])

#error = np.zeros(num_parameters)
#error[0] = abs((new_parat[0] - old_parat[0]) / new_parat[0])*100
#error[1] = abs((new_parat[1]- old_parat[1]) / new_parat[1])*100
#print(error[0])
#print(error[1])
#*****************************************************************
'''
