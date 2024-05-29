import solvers.cython.verle_cython as vc
import numpy as np

def sol_verle_c(y_0, interval, dt, mass):
    return vc.sol_verle_cython(y_0.astype(np.float64), interval, dt, mass.astype(np.float64))