import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as anim
import time

def animateSolarSystem(x, save_title): # x --- [[[x, y] * planet] * time] --- positions
    
    def update(i, x, scat):
        #print(i, x[i, 0], x[i, 1])
        scat.set_offsets(x[i])
        #scat.set_offsets(np.stack([x[i, :, 0], x[i, :, 1]]).T)
        return scat,
    
    #fig, ax = plt.subplots()
    #ax.set_xlim([x[0, 1, 0], x[0, 1, 0]])
    #ax.set_ylim([x[0, 1, 0], x[0, 1, 0]])
    
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    areas = [1000, 100, 150, 200, 180, 1000, 900, 800, 700]
    
    fig = plt.figure()
    
    #scat = ax.scatter(x[0, 0], x[0, 1], c=1, s=100)
    scat = plt.scatter(x[0, :, 0], x[0, :, 1], c=colors, s=areas)
    #scat = plt.scatter(x[1, :, 0], x[1, :, 1], c=colors, s=areas)
    #plt.show()
    
    
    #xlim, ylim
    print('ready')
    ani = FuncAnimation(fig, update, frames = len(x) - 1, fargs=(x, scat), interval=150, blit=True, repeat=True)
    writer = anim.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_title, writer=writer)
    plt.show()
    # ani ....
    #plt.close()

def calc_error(sol1, sol2, sol3, sol4, sol5, sol6, t):
    titles = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupyter', 'Saturn', 'Uran', 'Neptune']
    solutions = [sol2, sol3, sol4, sol5, sol6]
    labels = ['verle', 'verle_mult', 'cython', 'opencl', 'gpu']
    fig, axs = plt.subplots(1, 8)
    fig.suptitle('Mean square error')
    #x = np.arange(2, 6)
    colors = ['r', 'g', 'b', 'm', 'c']
    
    for i in range(len(axs)):
        y = [np.linalg.norm(sol1[:, i, :] - sol_i[:, i, :], axis=1) for sol_i in solutions]
        for j in range(len(y)):
            axs[i].plot(t, y[j], c=colors[j], label=labels[j])
        axs[i].set_title(titles[i])
        axs[i].legend()
    
    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='error')
    

def calc_times(times, N):
    fig = plt.figure()
    fig.suptitle('Elapsed time')
    
    #x = [1, 2, 3]
    x_signs = [str(n) for n in N]
    colors = ['b', 'g', 'r', 'm', 'c']
    labels = ['verle', 'verle_mult', 'cython', 'opencl', 'gpu']
    
    for i in range(times.shape[0]):
        plt.plot(N, times[i], c=colors[i], label=labels[i])
    
    fig.legend()
    
    fig2 = plt.figure()
    fig2.suptitle('Acceleration')
    
    for i in range(times.shape[0]):
        plt.plot(N, times[0]/times[i], c=colors[i], label=labels[i])
    
    fig2.legend()
    