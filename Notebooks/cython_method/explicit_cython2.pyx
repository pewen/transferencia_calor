import cython
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:] explicit_cython(np.ndarray[np.float_t,ndim=1] u, float kappa, float dt, float dz, np.ndarray[np.float_t,ndim=1] term_const, 
                    unsigned int nz, np.ndarray[long,ndim=1] plot_time):
    '''Cython version of explicit method'''
    
    #Defining C types
    cdef unsigned int i, k, j
    cdef unsigned int len_plot = len(plot_time) - 1
    cdef float lamnda = kappa*dt/dz**2
    
    # Memoryview on a NumPy array
    cdef double [:] u_view = u
    cdef double [:] un_view = u
    cdef double [:] const_view = term_const
    cdef double [:,:] uOut_view = np.zeros([len_plot + 1, nz])
    cdef long [:] plot_view = plot_time

    uOut_view[0] = u_view
    
    for i in range(len_plot):
        for k in range(plot_view[i], plot_view[i+1]):
            un_view = u_view
            for j in range(1, nz-1):
                u_view[j] = un_view[j] + lamnda*(un_view[j+1] - 2*un_view[j] + un_view[j-1]) + const_view[j]
        uOut_view[i+1] = u_view
 
    return uOut_view