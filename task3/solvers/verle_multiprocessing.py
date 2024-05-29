import numpy as np 
import multiprocessing as mp

G = 6.674e-11

def fill_x(y_prev, dt):
    y = np.zeros(y_prev.shape)
    y[0] = y_prev[0] + y_prev[2]*dt + y_prev[4]*dt**2/2
    y[1] = y_prev[1] + y_prev[3]*dt + y_prev[5]*dt**2/2
    return y

def fill_a(y, mass, num_obj, i):
    a_i_new = np.zeros((2,))
    for j in range(0, num_obj):
        if i != j:
            dx = y[j, 0] - y[i, 0]
            dy = y[j, 1] - y[i, 1]
            cnst = G * mass[j] / ( np.power(np.sqrt(dx**2 + dy**2), 3) )
            a_i_new[0] += cnst * dx
            a_i_new[1] += cnst * dy
    return a_i_new

def fill_v(v_prev, a_prev, a_new, dt):
    v_new = np.zeros(v_prev.shape)
    v_new[0] = v_prev[0] + (a_new[0] + a_prev[0])*dt/2
    v_new[1] = v_prev[1] + (a_new[1] + a_prev[1])*dt/2
    return v_new

def sol_verle_mult(y_0, interval, dt, mass):
    # y_0 = [[x, y, v_x, v_y], ...]
    # len(y_0) = num_objects
    
    num_obj = len(y_0)
    y = np.zeros((len(interval), num_obj, 6)) # y = [[x, y, v_x, v_y, a_x, a_y], ...]
    y[0, :, :4] = y_0
    
    p = mp.Pool()
    
    y[0, 1:, 4:] = np.array(p.starmap(fill_a, [(y[0, :, 0:2], mass, num_obj, i) for i in range(1, num_obj)]))
    
    for t in range(1, len(interval)):
        
        #x
        y[t, 1:] = np.array(p.starmap(fill_x, [(y[t-1, i], dt) for i in range(1, num_obj)]))
        
        #a
        y[t, 1:, 4:] = np.array(p.starmap(fill_a, [(y[t, :, 0:2], mass, num_obj, i) for i in range(1, num_obj)]))
        
        #v
        v_new = p.starmap(fill_v, [(y[t-1, i, 2:4], y[t-1, i, 4:], y[t, i, 4:], dt) for i in range(1, num_obj)])
        y[t, 1:, 2:4] = np.array(v_new)
    
    p.close()
    p.join()
    
    return y[:, :, :4]