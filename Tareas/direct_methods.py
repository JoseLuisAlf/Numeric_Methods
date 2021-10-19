
import numpy as np
import math

#**********************************************************************

#******************* Sustitución directa (FORWARD) ********************
def sust_directa(A,B):
    
    n = len(A)
    sol = np.zeros(n)
    for i in range(n):
        sol[i]=B[i]/A[i][i]
        for j in range(i):
            sol[i]=sol[i]-A[i][j]*sol[j]/A[i][i]

    return sol
#**********************************************************************

#----------------------------   SUSTITUCIÓN INVERSA (BACKWARD)   -------------
def sust_inversa(A,B):

    n = len(A)
    sol = np.zeros(n)
    for i in range(n):
        sol[n-1-i]=B[n-i-1]/A[n-i-1][n-i-1]
        for j in range(n-i,n):
            sol[n-1-i]=sol[n-1-i]-A[n-1-i][j]*sol[j]/A[n-1-i][n-1-i]

    return sol
#**********************************************************************


#-----------    PIVOTEO PARCIAL ---------------------------------------------
def pivoteo_parcial(A):

    n = len(A)
    M = np.copy(A)
    Naux=np.zeros(n)
    for i in range(n):
        if M[i][i] == 0:
            aux=M[i][i]
            for j in range(i+1,n):
                if M[j][i]>aux:
                    aux=M[j][i]
                    index=j
            for l in range(n):
                Naux[l]=M[index][l]
                M[index][l]=M[i][l]
                M[i][l]=Naux[l]
    return M
#**********************************************************************

#------------------------   ELIMINACION DE GAUUS    --------------------------
def eliminacion_gauss(A,B):

    def pivoteo_parcial(A):

        n = len(A)
        M = np.copy(A)
        Naux=np.zeros(n)
        for i in range(n):
            if M[i][i] == 0:
                aux=M[i][i]
                for j in range(i+1,n):
                    if M[j][i]>aux:
                        aux=M[j][i]
                        index=j
                for l in range(n):
                    Naux[l]=M[index][l]
                    M[index][l]=M[i][l]
                    M[i][l]=Naux[l]
        return M

    def sust_inversa(A,B):

        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[n-1-i]=B[n-i-1]/A[n-i-1][n-i-1]
            for j in range(n-i,n):
                sol[n-1-i]=sol[n-1-i]-A[n-1-i][j]*sol[j]/A[n-1-i][n-1-i]

        return sol

    n = len(A)
    new_A=np.zeros((n,n+1))
    for i in range(n):
        new_A[i]=np.concatenate((A[i],[B[i]]))

    new_A = pivoteo_parcial(new_A)

    for i in range(0,n-1):
        for j in range(1,n-i):
            aux=[new_A[j+i][i]/new_A[i][i]*s for s in new_A[i]]
            #print(aux)
            for l in range(n+1):
                new_A[j+i][l]=new_A[j+i][l]-aux[l]

    new_B = np.zeros(n)
    for i in range(n):
        new_B[i] = new_A[i][n]

    new_A = np.delete(new_A, n, axis = 1)

    sol = sust_inversa(new_A, new_B)

    return sol
#**********************************************************************

#--------------------   CROUT   ---------------------------------------------
def lu_crout(A,B):

    def sust_directa(A,B):
    
        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[i]=B[i]/A[i][i]
            for j in range(i):
                sol[i]=sol[i]-A[i][j]*sol[j]/A[i][i]

        return sol

    def sust_inversa(A,B):

        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[n-1-i]=B[n-i-1]/A[n-i-1][n-i-1]
            for j in range(n-i,n):
                sol[n-1-i]=sol[n-1-i]-A[n-1-i][j]*sol[j]/A[n-1-i][n-1-i]

        return sol
    
    n = len(A)
    l=np.zeros((n,n))
    u=np.zeros((n,n))
    for i in range(n):
        l[i][i]=1
    for j in range(0,n):
        for i in range(0,j+1):
            u[i][j]=A[i][j]
            for k in range(0,i):
                u[i][j]=u[i][j]-l[i][k]*u[k][j]

        for i in range(j+1,n):
            l[i][j]=A[i][j]/u[j][j]
            for k in range(0,j):
                l[i][j]=l[i][j]-l[i][k]*u[k][j]/u[j][j]


    d = sust_directa(l, B)
    sol = sust_inversa(u, d)

    return sol




#**********************************************************************

#----------------   CHOLESKY    ---------------------------------------------
def lu_cholesky(A,B):

    def sust_directa(A,B):
    
        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[i]=B[i]/A[i][i]
            for j in range(i):
                sol[i]=sol[i]-A[i][j]*sol[j]/A[i][i]

        return sol

    def sust_inversa(A,B):

        n = len(A)
        sol = np.zeros(n)
        for i in range(n):
            sol[n-1-i]=B[n-i-1]/A[n-i-1][n-i-1]
            for j in range(n-i,n):
                sol[n-1-i]=sol[n-1-i]-A[n-1-i][j]*sol[j]/A[n-1-i][n-1-i]

        return sol

    n = len(A)
    l=np.zeros((n,n))
    for i in range(0,n):
        l[i][i]=A[i][i]
        for k in range(0,i):
            l[i][i]=l[i][i]-l[i][k]*l[i][k]
        l[i][i]=math.sqrt(l[i][i])
        for j in range(i+1,n):
            l[j][i]=A[j][i]/l[i][i]
            for r in range(0,i):
                l[j][i]=l[j][i]-l[j][r]*l[i][r]/l[i][i]

    u=np.zeros((n,n))
     
    for i in range(0,n):
        for j in range(0,n):
            u[i][j]=l[j][i]
    
    d = sust_directa(l, B)
    sol = sust_inversa(u, d)

    return sol
#**********************************************************************
