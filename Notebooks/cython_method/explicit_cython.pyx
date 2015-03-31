# explicit cython
#Only compile the pure python code to C.
def explicit_cython(u, kappa, dt, dz, term_const, nz, plot_time):
    '''Cython version of explicit method'''
    u_out = []
    
    u_out.append(u.copy())
    
    for i in range(len(plot_time) - 1):
        for k in range(plot_time[i], plot_time[i+1]):
            un = u.copy()
            for i in range(1, nz-1):
                u[i] = un[i] + kappa*dt/dz**2*(un[i+1] - 2*un[i] + un[i-1]) + term_const[i]
        u_out.append(u.copy())
        
    return u_out