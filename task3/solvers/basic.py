import numpy as np 
from scipy.integrate import odeint

G = 6.674e-11

class Basic:

    @staticmethod
    def basic(y, t, mass):
        # y = [x, y, v_x, v_y, ...]
        # len(y) = 4 * num_objects
    
        f = np.zeros(y.shape)
        y_len = y.shape[0] // 4
        
        for i in range(1, y_len):
            f[4*i] = y[4*i+2]
            f[4*i+1] = y[4*i+3]
            
        for i in range(1, y_len):
            for j in range(0, y_len):
                if i != j:
                    dx = y[4*j] - y[4*i]
                    dy = y[4*j+1] - y[4*i+1]
                    cnst = G * mass[j] / ( (dx**2 + dy**2)**(3/2) )
                    f[4*i+2] += cnst * dx
                    f[4*i+3] += cnst * dy
                        
        return f

def sol_basic(y, t, dt, m):
    combined_solution = odeint(Basic.basic, y.flatten(), t, args = (m,))
    vector_solution = combined_solution.reshape(len(t), -1, 4)
    #print(vector_solution)
    return vector_solution