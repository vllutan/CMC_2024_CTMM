import numpy as np
from solvers.basic import sol_basic
from solvers.verle import sol_verle
from solvers.verle_multiprocessing import sol_verle_mult
from solvers.cython.run_verle_cython import sol_verle_c
from solvers.verle_opencl import sol_verle_opencl
from solvers.verle_gpu import sol_verle_gpu
from draw import calc_times
import time

if __name__=="__main__":
    total_time_start = time.time()
    
    interval = np.arange(0, 15, 1)
    dt = interval[1] - interval[0]
    
    N = [100, 200, 400]
    rng = np.random.default_rng()
    particles = [np.array([rng.integers(low=0, high=1e10, size=5) for i in range(j)]) for j in N]
    #print(particles)
    
    methods = [sol_verle, sol_verle_mult, sol_verle_c, sol_verle_opencl, sol_verle_gpu]
    #print(methods)
    
    times = np.zeros((len(methods), len(particles)))
    for m, m_ind in zip(methods, range(len(methods))):
        for p, p_ind in zip(particles, range(len(particles))):
            t = 0
            for k in range(3):
                t_start = time.time()
                m(p[:, :4], interval, dt, p[:, 4])
                t_end = time.time()
                t += t_end - t_start
            times[m_ind, p_ind] = t/3
    print(times)
    
    total_time_end = time.time()
    print(f"Total time for execution of {N} took {total_time_end - total_time_start}")
    
    calc_times(times, N)