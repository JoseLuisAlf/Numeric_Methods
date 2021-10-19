#!/usr/bin/python3

from numpy import *
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math # Solo para importar funciones trigonométricas y la constante pi
from pylab import *
from CAMPO_modulo import *

#//////////////////////////     PREGUNTA 1      //////////////////////////////
print('//////////////////////////     PREGUNTA 1      //////////////////////////////')
print('\n')
print('b) Suponiendo un potencia V(x) = x^4, hallamos el periodo T para a = 2 y m = 1: ')
print('\n')

def f1(x, parametro): # En esta funcion no se usara el parametro introducido
    
    # Definimos los parametros:
    a = 2
    m = 1

    f = pow(8*m,0.5)/pow((pow(a,4) - pow(x,4)), 0.5)

    return f

def f2(x, a): # En esta funcion, el parametro 'a' irá variando
    
    # Definimos los parametros:
    m = 1

    f = pow(8*m,0.5)/pow((pow(a,4) - pow(x,4)), 0.5)

    return f
# Definimos los límites y parametros:

x0 = 0
xf = 2

sol = gauss_legendre(x0, xf, f1, 0, 5) # Como f1 no depende del parametro, entonces lo igualamos a cero
print('                     T = ', sol)

#-------------------------------------------------------

print('\n')
print('c) Construya la gráfica T vs a para a en [0;2]: ')
print('\n')
print('Para a = 0, tenemos que T es igual a Nan ( Not a number ). Para evitar esto, tomamos el punto 0.01')
print('\n')

# Definimos los límites del intervalo y a_vec
a0 = 0.01
af = 2

# Dominio de a
a_dom = np.linspace(a0, af, 100)

# Calculamos el rango de la gráfica
T = np.array([ gauss_legendre(x0, i, f2, i, 5) for i in a_dom])
print('Valores graficados para T:')
print('\n')
print(T)
print('\n')

# Graficamos

plt.rc('axes', titlesize = 24)
plt.title('$T\ (Periodo)\ vs\ a\ (amplitud)$')
plt.xlabel('$a\ (m)$', size = 22)
plt.ylabel('$T\ (s)\ -\ Escala\ log$', size = 22)
plt.plot(a_dom, T, 'k-')
plt.yscale('log')
plt.grid()
plt.show()

#//////////////////////////     PREGUNTA 2      //////////////////////////////
print('//////////////////////////     PREGUNTA 2      //////////////////////////////')
print('\n')
print('b) Considerando el siguiente sistema de ecuaciones diferenciales, resuelva usando el método RK4')
print('   con los parametros definidos en el problema: ')
print('\n')

def dw(t, f):

    G = 1
    M = 10  
    L = 2
    r = pow( pow(f[2],2) + pow(f[3],2), 0.5 )
    dw = -G*M*f[2]/( pow(r,2)*pow( pow(r,2) + pow(L,2)/4, 0.5 ) )

    return dw

def dg(t, f):

    G = 1
    M = 10
    L = 2
    r = pow( pow(f[2],2) + pow(f[3],2), 0.5 )
    dg = -G*M*f[3]/( pow(r,2)*pow( pow(r,2) + pow(L,2)/4, 0.5 ) )

    return dg

def dx(t, f):

    dx = f[0]

    return dx

def dy(t, f):

    dy = f[1]

    return dy

# Definimos los parametros que pasaremos a la función
t0, tf, step = 0, 10, 0.005
f0_vec = np.array([0,1,1,0])
equ_vector = np.array([dw, dg, dx, dy])

# Llamamos a la función
r_vec = s_ode_RK4(t0, tf, f0_vec, step, equ_vector)

# Graficamos
n_pts = len(r_vec)
x = np.array([r_vec[i][2] for i in range(n_pts)])
y = np.array([r_vec[i][3] for i in range(n_pts)])

plt.rc('axes', titlesize = 24)
plt.title('$Grafica\ de\ la\ orbita\ de\ la\ particula\ m$')
plt.xlabel('$y$', size = 24)
plt.ylabel('$x$', size = 24)
plt.plot(y, x, 'k-')
plt.grid()
plt.show()

#//////////////////////////     PREGUNTA 3      //////////////////////////////
print('//////////////////////////     PREGUNTA 3      //////////////////////////////')
print('\n')
print('a) Considerando los parametros definidos en la pregunta, calculamos n(x,t) para 10 valores')
print('   en el intervalo [0,5]. En nuestro caso tomamos los puntos:')
print('\n')
print('             Valores para t = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]')
print('\n')
print('Obteniendo el siguiente resultado:')

# Definimos los parametros del problema
L = 2
C = 1
D = C
step_t = 0.0005
#t_lim = 10000

# Definimos los argumentos para la llamada al módulo
a = -L/2
b = L/2
n_pts = 6
stepx = (b-a)/(n_pts-1)
s = D*step_t/pow(stepx,2)
fact = np.array([0.5*s, (1 + C*step_t/2 - s), (1 - C*step_t/2 + s)])
parametro = L
error = 0.00001
xdom = np.linspace(a,b, n_pts)

# Definimos la funcion para las condicines iniciales
def f(x, parametro):

    f = -1*pow(x,2) + pow(parametro,2)/4
    return f

# En esta matriz se guardaran los perfiles para cada tiempo escogido
n_vec1 = np.zeros((10,n_pts))

# Este vector servira para calcular los perfiles de n pedidos en cada t escogido
t_lim_vec = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]) 

# Resolvemos el sistema 
for i in range(10):
    
    t_lim = t_lim_vec[i]
    x0 = np.zeros( t_lim+1 )
    xf = np.zeros( t_lim+1 )
    n =  equ_parabolica(a, b, fact, n_pts, step_t, t_lim, f, x0, xf, parametro, error)
    n_vec1[i] = n[t_lim] # Después de cada calculo, escogemos la ultima fila de la matriz, pues es el
                         # perfil de n requerido   


# Calculamos los promedios

n_prom1 = np.zeros(10) # Aquí se guardaran los promedios de cada perfil

# Se calculan los promedios
for i in range(10):
    
    suma =0

    for j in range(n_pts):

        suma = suma + n_vec1[i][j]

    n_prom1[i] = suma/n_pts

# Linealizamos

tdom = np.linspace(0.5,5,10)
ln_nprom1 = np.log(n_prom1)
coef0_1, coef1_1 = linear_regression(tdom, ln_nprom1)
A1 = np.exp(coef0_1)
alpha1 = 1/coef1_1

def fexp(t, A, alpha):

    f = A*np.exp(t/alpha)
    return f

tdom_aux = np.linspace(0.5,5,100)
f_ajuste1 = np.array([fexp(t, A1, alpha1) for t in tdom_aux])

# Grafica para  L = 2
plt.rc('axes', titlesize = 24)
plt.title(r'$\bar{n}\ (Promedio)\ vs\ t\ (\ L\ =\ 2\ )$')
plt.xlabel('$t$', size = 24)
plt.ylabel(r'$\bar{n}$', size = 24)
plt.plot(tdom, n_prom1, 'bv-')
plt.grid()
plt.show()

#-------------------------------------------------------------------------

print('\n')
print('b) Repetimos los pasos del item, pero ahora consideramos L = 4')
 
# Definimos los parametros del problema
L = 4
C = 1
D = C
step_t = 0.0005
#t_lim = 10000

# Definimos los argumentos para la llamada al módulo
a = -L/2
b = L/2
n_pts = 5
stepx = (b-a)/(n_pts-1)
s = D*step_t/pow(stepx,2)
fact = np.array([0.5*s, (1 + C*step_t/2 - s), (1 - C*step_t/2 + s)])
parametro = L
error = 0.0001

def f(x, parametro):

    f = -1*pow(x,2) + pow(parametro,2)/4
    return f

n_vec2 = np.zeros((10,n_pts))

for i in range(10):
    
    t_lim = t_lim_vec[i]
    x0 = np.zeros( t_lim+1 )
    xf = np.zeros( t_lim+1 )
    n =  equ_parabolica(a, b, fact, n_pts, step_t, t_lim, f, x0, xf, parametro, error)
    n_vec2[i] = n[t_lim]

# Calculamos los promedios

n_prom2 = np.zeros(10)

for i in range(10):
    
    suma =0

    for j in range(n_pts):

        suma = suma + n_vec2[i][j]

    n_prom2[i] = suma/n_pts


# Linealizamos

ln_nprom2 = np.log(n_prom2)
coef0_2, coef1_2 = linear_regression(tdom, ln_nprom2)
A2 = np.exp(coef0_2)
alpha2 = 1/coef1_2

tdom_aux = np.linspace(0.5,5,100)
f_ajuste2 = np.array([fexp(t, A2, alpha2) for t in tdom_aux])


# Grafica para  L = 4
plt.rc('axes', titlesize = 24)
plt.title(r'$\bar{n}\ (Promedio)\ vs\ t\ (\ L\ =\ 4\ )$')
plt.xlabel('$t$', size = 24)
plt.ylabel(r'$\bar{n}$', size = 24)
plt.plot(tdom, n_prom2, 'rv-')
plt.grid()
plt.show()

# Graficamos los ajustes
print('\n')
print('Para los ajustes, tenemos que:')
print('\n')
print('                         Funcion modelo: ñ = Ae^(t/alpha)')
print('\n')
print('Linealizando:')
print('\n')
print('                             ln(ñ) = ln(A) + t/alpha')
print('\n')
print('Aplicando regresion lineal:')

# Graficamos los ajustes

plt.subplot(1,2,1)
plt.plot(tdom, n_prom1,'bo', tdom_aux, f_ajuste1, 'k-')
plt.rc('axes', titlesize = 20)
plt.title('$L\ =\ 2\ (\ alpha\ =\ {}\ )$'.format(alpha1))
plt.xlabel('$t$', size = 24)
plt.ylabel(r'$\bar{n}$', size = 24)
plt.legend(('Promedios calculados', 'Ajuste de curva'), prop = {'size': 20}, loc='upper right')
plt.grid()

plt.subplot(1,2,2)
plt.plot(tdom, n_prom2, 'ro', tdom_aux, f_ajuste2, 'k-')
plt.rc('axes', titlesize = 20)
plt.title('$L\ =\ 4\ (\ alpha\ =\ {}\ )$'.format(alpha2))
plt.text(10,5, 'alpha = {}'.format(1/coef1_2))
plt.xlabel('$t$', size = 24)
plt.ylabel(r'$\bar{n}$', size = 24)
plt.legend(('Promedios calculados', 'Ajuste de curva'), prop = {'size': 20}, loc='upper left')
plt.grid()
plt.show()

print('\n')
print('c) ¿ Son las graficas ñ vs t las mismas para L = 2 y L = 4? ¿ Puede extraer alguna conclusion ')
print('     de los calculos anteriores')
print('\n')
print('Podemos observar que ambas graficas son opuestas, en el sentido de que para L = 2, tenemos un decaimiento')
print('en la densidad de neutrones a medida que transcurre el tiempo. Si embargo, para L = 4, tenemos que este')
print('aumenta. Entre los valores 2 y 4, existe algun valor para L en el cual podemos observar el inicio de este') 
print('cambio en la densidad, como un punto en el que la cantidad de neutrones empieza a aumentar (como un proceso')
print('en cadena que aumenta la produccion de neutrones).')

