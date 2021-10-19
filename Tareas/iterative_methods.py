
import numpy as np
import math

#************************* MÉTODOS ITERATIVOS ****************************

#************************ Metodo Gauss - Seidel **************************
def gauss_seidel(A, B, accurate):
    
    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    size = len(A)
    iterations=0
    error = np.array([])
    x=np.zeros(size)
    xold=np.zeros(size)
    bolean=True
    while bolean :
        for i in range(size):
            xold[i]=x[i]
        for i in range(size):
            x[i]=B[i]/A[i][i]
            for j in range(i+1,size):
                x[i]=x[i]-A[i][j]*x[j]/A[i][i]
            for j in range(0,i):
                x[i]=x[i]-A[i][j]*x[j]/A[i][i]
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold) < accurate:
            bolean=False
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones
#*************************************************************************

#****************************** Método de Jacobi *************************
def jacobi(A, B, accurate):
    
    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    size = len(A)
    iterations=0
    error = np.array([])
    x=np.zeros(size)
    xold=np.zeros(size)
    bolean=True
    aux=np.zeros(size)
    while bolean:
        for i in range(size):
            xold[i]=x[i]
        for i in range(size):
            x[i]=B[i]/A[i][i]
            for j in range(i+1,size):
                x[i]=x[i]-A[i][j]*aux[j]/A[i][i]
            for j in range(0,i):
                x[i]=x[i]-A[i][j]*aux[j]/A[i][i]
        aux=x
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold)<accurate:
            bolean=False
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones
#*************************************************************************

#*****************************  SOR **************************************
def sor(A, B, omega, accurate):

    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    size = len(A)
    iterations=0
    error = np.array([])
    x=np.zeros(size)
    xold=np.zeros(size)
    bolean=True
    while bolean :
        for i in range(size):
            xold[i]=x[i]
        for i in range(size):
            x[i]=(1-omega)*x[i]+omega*B[i]/A[i][i]
            for j in range(0,i):
                x[i]=x[i]-omega*A[i][j]*x[j]/A[i][i]
            for j in range(i+1,size):
                x[i]=x[i]-omega*A[i][j]*x[j]/A[i][i]
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold)<accurate:
            bolean=False
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones
#*************************************************************************

#****************** Máximo Descenso **********************************
def maximum_descent(A, B, accurate):

    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    def grad(x,A,b):    # Residuo para el metodo de gradiente conjugado
        r=np.dot(A,x)-b    
        return r

    def Alpha(x,A,b):   # Alpha para el metodo de maximo descenso
        alpha=np.dot(grad(x,A,b),grad(x,A,b))/np.dot(grad(x,A,b),np.dot(A,grad(x,A,b)))
        return alpha

    size = len(A)
    x=np.zeros(size)
    iterations = 0
    error = np.array([])
    bolean=True
    while bolean :
        xold=x
        x=x-Alpha(x,A,B)*grad(x,A,B)
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold)<accurate:
            bolean=False
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones
#*********************************************************************

#****************** Gradiente Conjugado **********************************
def conjugate_gradient(A, B, accurate):

    def distancia(size,vector1,vector2):    # Funcion que me permite hallar la distancia entre dos puntos
        d=0                                 # Definida para calcular el error
        for i in range(size):
            d=d+(vector1[i]-vector2[i])**2
        d=math.sqrt(d)
        return d

    size = len(A)
    error = np.array([])
    x=np.zeros(size)
    p=np.zeros(size)
    r=np.zeros(size)
    raux=np.zeros(size)
    iterations = 0
    bolean=True
    r=np.dot(A,x)-B
    p=-1*r
    betha=0
    while bolean :
        xold=x
        raux=r
        alpha=np.dot(r,r)/np.dot(p,np.dot(A,p))
        x=x+alpha*p
        r=r+alpha*np.dot(A,p)
        betha=np.dot(r,r)/np.dot(raux,raux)
        p=-r+betha*p
        error = np.append(error, distancia(size,x,xold))
        iterations += 1
        if distancia(size,x,xold)<0.00001:
            bolean=False
    return x, error, iterations # Devuelve: Sol, errores relativos, iteraciones
#*************************************************************************



