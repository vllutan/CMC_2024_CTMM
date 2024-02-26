import numpy as np
import math
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import json

class Mesh:
    def __init__(self, filepath):
        self.vertices = self.loadMesh(filepath)
        self.num_objects = len(self.vertices)
        print("Number of objects", self.num_objects, ", number of faces", [len(x)/9 for x in self.vertices])
        #for i in range(self.num_objects):
        #    print("Max vertices for object", i)
        #    print("max x", max([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 0]))
        #    print("max y", max([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 1]))
        #    print("max z", max([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 2]))
        #    print("min x", min([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 0]))
        #    print("min y", min([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 1]))
        #    print("min z", min([self.vertices[i][x] for x in range(len(self.vertices[i])) if x%3 == 2]))
        self.s_i, self.s_ij = self.countArea()
        
    def loadMesh(self, filepath):
        vertices = []
        obj_vertices = []
        v = []
        index = []
        
        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                
                if flag == "v":
                    line = line.replace("v ", "").strip()
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "f":
                    line = line.replace("f ", "").strip()
                    line = line.split(" ")
                    faceVertices = []
                    for vert in line:
                        faceVertices.append(v[int(vert) - 1])
                    for i in [0,1,2]:
                        for x in faceVertices[i]:
                            obj_vertices.append(x)
                elif flag == "g":
                    index.append(line[-2])
                    print(index)
                    if len(obj_vertices):
                        vertices.append(obj_vertices)
                    obj_vertices = []
                else:
                    pass
                
                line = f.readline()
        vertices.append(obj_vertices)
        vertices = [x for _, x in sorted(zip(index, vertices), key=lambda pair: pair[0])]
        return vertices
    
    def countArea(self):
        S_I = np.zeros((self.num_objects))
        S_IJ = np.zeros((self.num_objects, self.num_objects))
            
        # S[0] = pi*(r1 + r2)*l + pi*r1^2
        z_max = max([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 2]) 
        z_min = min([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 2])
        r_big = max([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 0])
        r_small = max([self.vertices[0][x] for x in range(len(self.vertices[0]) - 2) if self.vertices[0][x+2] == z_max and x%3 == 0])
        l = math.sqrt((z_max - z_min)**2 + (r_big - r_small)**2)
        
        S_I[0] = math.pi*((r_big + r_small)*l + r_small**2)
        
        # S[1] =  2*pi*r*h
        r = max([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 0])
        z_max = max([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 2]) 
        z_min = min([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 2])
        h = z_max - z_min
        
        S_I[1] = 2*math.pi*r*h
        S_IJ[1][2] = S_IJ[2][1] = math.pi*r**2
        S_IJ[1][0] = S_IJ[0][1] = math.pi*r**2
        
        # S[3] = 2*pi*r*h
        x_max = max([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 0]) 
        x_min = min([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 0])
        y_max = max([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 1]) 
        y_min = min([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 1])
        r = abs((x_max - x_min)/2)
        h = abs(y_max - y_min)
        
        S_I[3] = 2*math.pi*r*h
        S_IJ[3][2] = S_IJ[2][3] = math.pi*r**2
        S_IJ[3][4] = S_IJ[4][3] = math.pi*r**2
        
        # S[2] = 2(ab + bc + ac)
        x_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 0]) 
        x_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 0])
        y_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 1]) 
        y_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 1])
        z_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 2]) 
        z_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 2])
        a = abs(x_max - x_min)
        b = abs(y_max - y_min)
        c = abs(z_max - z_min)
        S_I[2] = 2*(a*b + b*c + a*c) - S_I[1] - S_I[3]
        
        # S[4] 
        x_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 0]) 
        x_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 0])
        y_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 1]) 
        y_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 1])
        z_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 2]) 
        z_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 2])
        z_mid = max([self.vertices[4][x] for x in range(2, len(self.vertices[4])) if self.vertices[4][x-1] == y_min and x%3 == 2])
        
        S_I[4] = abs((x_max - x_min)*(y_max - y_min)) + abs((y_max - y_min)*(z_max + z_mid - 2*z_min)) + \
            abs((x_max - x_min)*(z_max - z_min)) + abs((x_max - x_min)* (z_mid - z_min)) + \
            abs((x_max - x_min)*math.sqrt((z_max - z_mid)**2 + (y_max - y_min)**2)) - S_I[3]
        
        print(S_I, "\n", S_IJ)
        
        return S_I, S_IJ

def coef_init(filepath):
    global eps, c, Q_R, lmbd
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
    #eps = [0.05, 0.05, 0.1, 0.01, 0.1]
    #c = [520, 520, 900, 840, 900]
    #lmbd = [[0, 20, 0, 0, 0], [20, 0, 130, 0, 0], [0, 130, 0, 10.5, 0], [0, 0, 10.5, 0, 119], [0, 0, 0, 119, 0]]

def temp_init(filepath):
    global T
    with open(filepath) as f:
        coefs = json.load(f)
    for name, value in coefs.items():
        if name == "T":
            T = value
        else:
            print("ERROR in json file", filepath)

def g0(y, LS, EpSC, A, c):
    #print(type(y))
    y_sub = np.array([y_min - y_def for y_min in y for y_def in y]).reshape(len(y), len(y))
    f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
    f[0] += 23*A/c[0]
    return f

def g1(y, t, LS, EpSC, A, c):
    #print(type(y))
    y_sub = np.array([y_min - y_def for y_min in y for y_def in y]).reshape(len(y), len(y))
    f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
    f[0] += A*(20 + 3*math.cos(t/4))/c[0]
    return f

def g2(t, y, LS, EpSC, A, c):
    #print(type(y))
    y_sub = np.array([y_min - y_def for y_min in y for y_def in y]).reshape(len(y), len(y))
    f = (- np.diag(LS @ y_sub) - EpSC * (y/100)**4)/c
    f[0] += A*(20 + 3*math.cos(t/4))/c[0]
    return f

C0 = 5.67
m = Mesh("model2.obj")
s_i = m.s_i
s_ij = m.s_ij
coef_init("coefficients.json")
LS = s_ij*lmbd
print(LS)
EpSC = eps*s_i*C0
print(EpSC)

init = [50, 50, 50, 50, 51]
root = fsolve(g0, init, args=(LS, EpSC, 1, c))
t = np.linspace(0, 10, 101)
sol1 = odeint(g1, root, t, args=(LS, EpSC, 1, c))
sol2 = solve_ivp(g2, [0, 10], root, args=(LS, EpSC, 1, c), t_eval=t)
print(root, sol1.shape, sol2.t.shape, sol2.y.shape, t.shape, sol2)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(t, sol1[:, 0])
ax.set_title("odeint")
 
ax = fig.add_subplot(1, 2, 2)
ax.plot(sol2.t, sol2.y[0, :])
ax.set_title("solve_ivp")