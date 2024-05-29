import numpy as np 
cimport numpy as cnp
cnp.import_array()
import cython
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt, pow

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t G = 6.674e-11

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=3] sol_verle_cython(cnp.ndarray[DTYPE_t, ndim=2] y_0, cnp.ndarray[int] interval, int dt, cnp.ndarray[DTYPE_t] mass):
    assert y_0.dtype == DTYPE and mass.dtype == DTYPE
    
    cdef Py_ssize_t num_obj = len(y_0), i, j, t
    cdef DTYPE_t dx, dy, cnst, temp
    cdef cnp.ndarray[DTYPE_t, ndim=3] y = np.zeros((len(interval), num_obj, 6), dtype = DTYPE) 
    y[0, :, :4] = y_0
    
    
    for i in prange(1, num_obj, nogil=True):
        for j in range(0, num_obj):
            if i != j:
                dx = y[0, j, 0] - y[0, i, 0]
                dy = y[0, j, 1] - y[0, i, 1]
                cnst = G * mass[j] / ( pow(sqrt(dx**2 + dy**2), 3) )
                y[0, i, 4] += cnst * dx
                y[0, i, 5] += cnst * dy
    
    for t in range(1, len(interval)):
        #x
        for i in prange(1, num_obj, nogil=True):
            y[t, i, 0] = y[t-1, i, 0] + y[t-1, i, 2]*dt + y[t-1, i, 4]*dt**2/2
            y[t, i, 1] = y[t-1, i, 1] + y[t-1, i, 3]*dt + y[t-1, i, 5]*dt**2/2
            
        #a
        for i in prange(1, num_obj, nogil=True):
            for j in range(0, num_obj):
                if i != j:
                    dx = y[t, j, 0] - y[t, i, 0]
                    dy = y[t, j, 1] - y[t, i, 1]
                    cnst = G * mass[j] / ( pow(sqrt(dx**2 + dy**2), 3) )
                    y[t, i, 4] += cnst * dx
                    y[t, i, 5] += cnst * dy
        
        #v
        for i in prange(1, num_obj, nogil=True):
            y[t, i, 2] = y[t-1, i, 2] + (y[t-1, i, 4] + y[t, i, 4])*dt/2
            y[t, i, 3] = y[t-1, i, 3] + (y[t-1, i, 5] + y[t, i, 5])*dt/2
    
    
    return y[:, :, :4]