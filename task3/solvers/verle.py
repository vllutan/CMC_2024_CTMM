import numpy as np 

G = 6.674e-11

def sol_verle(y_0, interval, dt, mass):
    # y_0 = [[x, y, v_x, v_y], ...]
    # len(y_0) = num_objects
    
    num_obj = len(y_0)
    y = np.zeros((len(interval), num_obj, 6)) # y = [[x, y, v_x, v_y, a_x, a_y], ...]
    y[0, :, :4] = y_0
    
    for i in range(1, num_obj):
        for j in range(0, num_obj):
            if i != j:
                dx = y[0, j, 0] - y[0, i, 0]
                dy = y[0, j, 1] - y[0, i, 1]
                cnst = G * mass[j] / ( (dx**2 + dy**2)**(3/2) )
                y[0, i, 4] += cnst * dx
                y[0, i, 5] += cnst * dy
    
    for t in range(1, len(interval)):
        #x
        for i in range(1, num_obj):
            y[t, i, 0] = y[t-1, i, 0] + y[t-1, i, 2]*dt + y[t-1, i, 4]*dt**2/2
            y[t, i, 1] = y[t-1, i, 1] + y[t-1, i, 3]*dt + y[t-1, i, 5]*dt**2/2
            
        #a
        for i in range(1, num_obj):
            for j in range(0, num_obj):
                if i != j:
                    dx = y[t, j, 0] - y[t, i, 0]
                    dy = y[t, j, 1] - y[t, i, 1]
                    cnst = G * mass[j] / ( (dx**2 + dy**2)**(3/2) )
                    y[t, i, 4] += cnst * dx
                    y[t, i, 5] += cnst * dy
        
        #v
        for i in range(1, num_obj):
            y[t, i, 2] = y[t-1, i, 2] + (y[t-1, i, 4] + y[t, i, 4])*dt/2
            y[t, i, 3] = y[t-1, i, 3] + (y[t-1, i, 5] + y[t, i, 5])*dt/2
    
    
    return y[:, :, :]