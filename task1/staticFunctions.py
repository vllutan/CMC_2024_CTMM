import numpy as np
import math
import json

class cstFunctions:
    
    # for fsolve, general version
    @staticmethod
    def g0(y, LS, EpSC, A, c):
        y_sub = np.array([y_def - y_min for y_min in y for y_def in y]).reshape(len(y), len(y))
        f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
        f[0] += 23*A/c[0]
        return f
    
    #for odeint, general version
    @staticmethod
    def g1(y, t, LS, EpSC, A, c):
        y_sub = np.array([y_def - y_min for y_min in y for y_def in y]).reshape(len(y), len(y))
        f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
        f[0] += A*(20 + 3*math.cos(t/4))/c[0]
        return f
    
    #for solve_ivp, general version
    @staticmethod
    def g2(t, y, LS, EpSC, A, c):
        y_sub = np.array([y_def - y_min for y_min in y for y_def in y]).reshape(len(y), len(y))
        f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
        f[0] += A*(20 + 3*math.cos(t/4))/c[0]
        return f
    
    # for fsolve, explicit version
    @staticmethod
    def g0_explicit(y, LS, EpSC, A, c):
        f0 = (                         - LS[0][1]*(y[0] - y[1]) - EpSC[0] * (y[0]/100)**4)/c[0]
        f1 = (- LS[1][0]*(y[1] - y[0]) - LS[1][2]*(y[1] - y[2]) - EpSC[1] * (y[1]/100)**4)/c[1]
        f2 = (- LS[2][1]*(y[2] - y[1]) - LS[2][3]*(y[2] - y[3]) - EpSC[2] * (y[2]/100)**4)/c[2]
        f3 = (- LS[3][2]*(y[3] - y[2]) - LS[3][4]*(y[3] - y[4]) - EpSC[3] * (y[3]/100)**4)/c[3]
        f4 = (- LS[4][3]*(y[4] - y[3])                          - EpSC[4] * (y[4]/100)**4)/c[4]
        f0 += A*23/c[0]
        return [f0, f1, f2, f3, f4]

    # for odeint, explicit version
    @staticmethod
    def g1_explicit(y, t, LS, EpSC, A, c):
        f0 = (                         - LS[0][1]*(y[0] - y[1]) - EpSC[0] * (y[0]/100)**4)/c[0]
        f1 = (- LS[1][0]*(y[1] - y[0]) - LS[1][2]*(y[1] - y[2]) - EpSC[1] * (y[1]/100)**4)/c[1]
        f2 = (- LS[2][1]*(y[2] - y[1]) - LS[2][3]*(y[2] - y[3]) - EpSC[2] * (y[2]/100)**4)/c[2]
        f3 = (- LS[3][2]*(y[3] - y[2]) - LS[3][4]*(y[3] - y[4]) - EpSC[3] * (y[3]/100)**4)/c[3]
        f4 = (- LS[4][3]*(y[4] - y[3])                          - EpSC[4] * (y[4]/100)**4)/c[4]
        f0 += A*(20 + 3*math.cos(t/4))/c[0]
        return [f0, f1, f2, f3, f4]
    
    # for solve_ivp, explicit version
    @staticmethod
    def g2_explicit(t, y, LS, EpSC, A, c):
        f0 = (                         - LS[0][1]*(y[0] - y[1]) - EpSC[0] * (y[0]/100)**4)/c[0]
        f1 = (- LS[1][0]*(y[1] - y[0]) - LS[1][2]*(y[1] - y[2]) - EpSC[1] * (y[1]/100)**4)/c[1]
        f2 = (- LS[2][1]*(y[2] - y[1]) - LS[2][3]*(y[2] - y[3]) - EpSC[2] * (y[2]/100)**4)/c[2]
        f3 = (- LS[3][2]*(y[3] - y[2]) - LS[3][4]*(y[3] - y[4]) - EpSC[3] * (y[3]/100)**4)/c[3]
        f4 = (- LS[4][3]*(y[4] - y[3])                          - EpSC[4] * (y[4]/100)**4)/c[4]
        f0 += A*(20 + 3*math.cos(t/4))/c[0]
        return [f0, f1, f2, f3, f4]
    
    
    @staticmethod 
    def coef_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "c":
                c = value
            elif name == "eps":
                eps = value
            elif name == "lmbd":
                lmbd = value
            else:
                print("ERROR in json file", filepath)
        return c, eps, lmbd
    
    @staticmethod
    def temp_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "T":
                T = value
            else:
                print("ERROR in json file", filepath)
        return T
    
    @staticmethod
    def time_init(filepath):
        with open(filepath) as f:
            coefs = json.load(f)
        for name, value in coefs.items():
            if name == "time_interval":
                t = np.linspace(0, value, 1001)
                v = value
            else:
                print("ERROR in json file", filepath)
        return t, v
    

"""
#sol2 = solve_ivp(g2, [0, 10], root, args=(LS, EpSC, 1, c), t_eval=t)
#print(root, sol1.shape, sol2.t.shape, sol2.y.shape, t.shape, sol2)
"""