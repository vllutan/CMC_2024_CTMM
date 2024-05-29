import numpy as np
from solvers.basic import sol_basic
from solvers.verle import sol_verle
from solvers.verle_multiprocessing import sol_verle_mult
from solvers.cython.run_verle_cython import sol_verle_c
from solvers.verle_opencl import sol_verle_opencl
from solvers.verle_gpu import sol_verle_gpu
from draw import animateSolarSystem, calc_error

if __name__=="__main__":
    #t = np.arange(0, 15 * 24*60*60, 5 * 24*60*60)
    t = np.arange(0, 1 * 60*60, 1)
    dt = t[1] - t[0]
    
    solar_system = np.zeros((9, 5)) # [x_0, y_0, v_x_0, v_y_0, m]

    solar_system[0] = [0, 0, 0, 0, 199e28] #sun
    solar_system[1] = [50e9, 0, 0, 47e3, 33e22] #mercury
    solar_system[2] = [108e9, 0, 0, 35e3, 487e22] #venus
    solar_system[3] = [15e7, 0, 0, 3e4, 5.97e24] #earth
    solar_system[4] = [228e6, 0, 0, 24.1e3, 6.42e23] #mars
    solar_system[5] = [778.5e6, 0, 0, 13.1e3, 1.8986e27] #Jupiter 
    solar_system[6] = [1.4e9, 0, 0, 9.69e3, 5.68e26] #saturn
    solar_system[7] = [2.8e9, 0, 0, 6.8e3, 8.68e25] #uran
    solar_system[8] = [4.55e9, 0, 0, 5.44e3, 3.33e23] #neptune
    
    sol1 = sol_basic(solar_system[:, :4], t, dt, solar_system[:, 4])
    #print('1\n', sol1[:, :, :2])
    #animateSolarSystem(sol1[:, :, :2], 'sol1.gif')
    
    sol2 = sol_verle(solar_system[:, :4], t, dt, solar_system[:, 4])
    #print('2\n', sol2[:, :, :2])
    #animateSolarSystem(sol2[:, :, :2], 'sol2.gif')
    
    sol3 = sol_verle_mult(solar_system[:, :4], t, dt, solar_system[:, 4])
    #print('3\n', sol3[:, :, :2])
    #animateSolarSystem(sol3[:, :, :2], 'sol3.gif')
    
    sol4 = sol_verle_c(solar_system[:, :4], t, dt, solar_system[:, 4])
    #print('4\n', sol4[:, :, :2])
    #animateSolarSystem(sol4[:, :, :2], 'sol4.gif')
    
    sol5 = sol_verle_opencl(solar_system[:, :4], t, dt, solar_system[:, 4])
    #print('5\n', sol5[:, :, :2])
    #animateSolarSystem(sol5[:, :, :2], 'sol5.gif')
    
    sol6 = sol_verle_gpu(solar_system[:, :4], t, dt, solar_system[:, 4])
    
    #print(np.linalg.norm(res_np - (a_np + b_np)))
    calc_error(sol1[:, :, :2], sol2[:, :, :2], sol3[:, :, :2], sol4[:, :, :2], sol5[:, :, :2], sol6[:, :, :2], t)
    