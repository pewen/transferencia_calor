import numpy as np

#To solve implicit method
from scipy import sparse as sp
from scipy.sparse import linalg as splinalg
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
#para cambiar el eje y 
from matplotlib import ticker
#Defino algunos parametros del plot (tamaño de los label, titulo, etc)
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20
rcParams['axes.labelsize']=25
rcParams['axes.titlesize']=25
rcParams['legend.fontsize']=20

from numba import autojit

#Metodo explicito usando cython
from cython_method import explicit_cython
from cython_method import explicit_cython2
from cython_method import explicit_cython3

#################################################################################
#Metodos explicitos
#################################################################################

def explicit_py(u, kappa, dt, dz, const, nz):
    un = u.copy()
    for i in range(1, nz-1):
        u[i] = un[i] + kappa*dt/dz**2*(un[i+1] - 2*un[i] + un[i-1]) + const[i]
    return u

@autojit
def explicit_numba(u, kappa, dt, dz, const, nz):
    un = u.copy()
    for i in range(1, nz-1):
        u[i] = un[i] + kappa*dt/dz**2*(un[i+1] - 2*un[i] + un[i-1]) + const[i]
    return u

def explicit_slicing(u, kappa, dt, dz, const):
    un = u.copy()
    u[1:-1] = un[1:-1] + kappa*dt/dz**2*(un[2:] - 2*un[1:-1] + un[:-2]) + const[1:-1]
    return u

#################################################################################
#Metodos Implicitos
#################################################################################
def implicit_scipy(u, kappa, dt, dz, T0, const, nz, plot_time):
    #Matriz tridiagonal
    lamnda = ( kappa * dt ) / dz**2
    diag_central = np.array([1 + 2 * lamnda for i in range(nz-2)])
    diag_sup = np.array([-lamnda for i in range(nz-2)])
    A = sp.dia_matrix(([diag_central, diag_sup, diag_sup], [0, 1, -1]), shape=(nz-2, nz-2))
    A = A.tocsc()

    #La factorizo en una LU
    A_LU  = splinalg.splu(A)
    
    #Vector b solucion
    b = np.zeros(nz-2)
    b_out = np.zeros(nz)
    b_out[-1] = T0
    u_out = []
    u_out.append(b_out.copy())
    
    for i in range(len(plot_time) - 1):
        for k in range(plot_time[i], plot_time[i+1]):
            #Condicion de contorno
            b[-1] = b[-1] + lamnda*T0
            #Le sumo el vector de constantes
            b = b + const[1:-1]
            b = A_LU.solve(b)
        b_out[1:-1] = b
        u_out.append(b_out.copy())
        
    return u_out

#################################################################################
#Solve. 
#################################################################################
def solve_explicit(z_total=35000, dz=1750, t_total=3.1536e14, dt=3.1536e8, T0=727, 
                  kappa=1e-6, A0=2e-6, p=5500, C=1260, L=10000, CFL=0.5, 
                  amount_plot = 2, metodo = 'explicit_slicing'):
    '''Solucion numerica de la ecuacion de difusion de calor
    
    Los numero de puntos en la grilla temporal (nt) son calculados teniendo en cuenta que se 
    cumpla la ecuacion de estabilidad
    
    Parameters
    ----------
    z_total  : integer
        profundidad total. [z] = m.
    dz : float
        Paso en la dimencion Z. [dz] = m.
    t_total : float
        tiempo total de calculo. [t_total] = s.
    dt : float
        Paso en el tiempo. De no cumplir con la ecuacion de estavilidad, 
          se lo recalculara. [dt] = s.
    T0 : float
        Temperatura en z. [T0] = K.
    kappa : float
        Difusion termica. [kappa] = m2/s.
    A0 : float
        Volumetric heat production rate. [A0] = W/m3
    p : float
        Densidad. [p] = m3/Kg
    C : float
        Calor especifico. [C] = J/(Kg K)
    L : float
        Longitud caracteristica. [L] = m
    CFL: float
        Valor para la estavilidad de la ecuacion
    amount_plot : int
        Cantidad de plots x-espaciados
    metodo: str
        metodo con el cual se va a resolver la ED
        
    Returns
    -------
    u : array
        Array solucion del problema.
    prop: dictionary
        Algunas propiedades del problema (dz, dt, z, t_total, nx, nt)
    '''
    
    #Calculo de cantidad de puntos en la dimension Z (nz) como z_total/dz.
    #Chequeo que nz sea entero. De no serlo, recalculo z_total.
    if z_total%dz == 0:
        nz = z_total//dz
    else:
        z_total = z_total + (dz - z_total%dz)
        nz = z_total//dz
    
    #Chequeo que el dt ingresado cumpla con la ecuacion de estabilidad
    #Si no lo cumple, se lo recalcula
    condicion_cfl = dz**2/(2*kappa)
    if dt > condicion_cfl:
        dt = condicion_cfl
    
    #Calculo de cantidad de puntos en la dimension temporal (nt) como t_total/dt.
    #Chequeo que nt sea entero. De no serlo, recalculo t_total.
    if t_total%dt == 0:
        nt = int(t_total//dt)
    else:
        t_total = t_total + (dt - t_total%dt)
        nt = int(t_total//dt)
    
    #Inicializacion de los array
    #Array de espacio
    z = np.array(range(0, z_total, dz))

    #Array solucion de la temp.
    u = np.zeros(nz)
    un = np.zeros(nz)
    #Condicion inicial
    u[-1] = T0
    
    #Como el ultimo termino del modelo no varia con el timepo
    #lo calculo antes y luego solo lo sumo vectorialmente
    const = (dt*A0)/(p*C)*np.exp(-z/L)
    
    prop = {}
    prop['dz'] = dz
    prop['dt'] = dt
    prop['z'] = z_total
    prop['t_total'] = t_total
    prop['nz'] = nz
    prop['nt'] = nt
    
    u_out = []
    
    
    if amount_plot <= 2:
        amount_plot = 2

    prop['amount_plot'] = amount_plot + 1
        
    plot_time = [int(i) for i in np.linspace(0, nt, amount_plot+1)]

    u_out.append(u.copy())
    
    if metodo == 'explicit_py':
        for i in range(len(plot_time) - 1):
            for k in range(plot_time[i], plot_time[i+1]):
                u = explicit_py(u, kappa, dt, dz, const, nz)
            u_out.append(u.copy())
    
    elif metodo == 'explicit_numba':
        for i in range(len(plot_time) - 1):
            for k in range(plot_time[i], plot_time[i+1]):
                u = explicit_numba(u, kappa, dt, dz, const, nz)
            u_out.append(u.copy())
            
    elif metodo == 'explicit_cython':
        k = explicit_cython.explicit_cython(u, kappa, dt, dz, const, nz, plot_time)
        return k, prop
    
    elif metodo == 'explicit_cython2':
        k = explicit_cython2.explicit_cython(u, kappa, dt, dz, const, nz, np.array(plot_time))
        return k, prop
    
    elif metodo == 'explicit_cython3':
        k = explicit_cython3.explicit_cython(u, kappa, dt, dz, const, nz, np.array(plot_time))
        return k, prop
    
    elif metodo == 'explicit_slicing':
        for i in range(len(plot_time) - 1):
            for k in range(plot_time[i], plot_time[i+1]):
                u = explicit_slicing(u, kappa, dt, dz, const)
            u_out.append(u.copy())
            
    elif metodo == 'implicit_scipy':
        u_out = implicit_py(u, kappa, dt, dz, T0, const, nz, plot_time)
    
    else:
        print('No existe ese metodo')
        return('No existe ese metodo')
    
    return u_out, prop

#################################################################################
#Pretty Plot
#################################################################################
def pretty_plot(T_out, prop, marker_plot = '-'):
    ''' Pretty plot to show solution of solve_explicit
    
    Parameters
    ----------
    T_out : array
            Arrays de salida de solve_explicito
    prop : dictionary
            Diccionario con algunas propiedades de la solucion numorica (dt, dz, z_total, etc)
            
    marker_plot : str
            Marcador para hacer el plot
            
    Returns
    -------
    Plot
    
    '''

    z = np.array(range(0, prop['z'], prop['dz']))

    fig, ax = plt.subplots(figsize=(12, 11))

    for i in range(len(T_out)):
        ax.plot(T_out[i], z, marker_plot, label=r'$time=%.2e$' % (prop['dt']*prop['nt'] * i))

    plt.xlabel('$T(K)$', fontsize=25)
    plt.ylabel(r'$z(m)$', fontsize=25)
    plt.title('$Heat\, conductions\; \Delta t=%.2e\; \Delta z=%.2e$' % \
              (prop['dt'], prop['dz']), fontsize=25)
    plt.legend(loc = "best", fontsize=20)

    #Para setear que el eje y sea en notacion cientifica
    ax.set_yticks([i for i in range(0, 40000, 5000)])

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter)

    plt.grid(True)