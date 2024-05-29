import numpy as np 
import pyopencl as cl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

G = 6.674e-11

def sol_verle_opencl(y_0, interval, dt, mass):
    # y_0 = [[x, y, v_x, v_y], ...]
    # len(y_0) = num_objects
    
    context = cl.Context(devices=cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU))
    #context = cl.Context()
    queue = cl.CommandQueue(context)
    
    program_fill_x = cl.Program(context, '''
                                __kernel void fill_x(__global const double *y_prev, __global double *y, const int dt, const int size)
                                {
                                    int i = get_global_id(0);
                                    
                                    y[i*2] = y_prev[i*size] + y_prev[i*size+2]*dt + y_prev[i*size+4]*dt*dt/2;
                                    y[i*2 + 1] = y_prev[i*size+1] + y_prev[i*size+3]*dt + y_prev[i*size+5]*dt*dt/2;
                                }''').build()
    
    program_fill_a = cl.Program(context, '''
                                __kernel void fill_a(const int num_obj, __global const double *mass, __global const double *y, __global double *a)
                                {
                                    int i = get_global_id(0) + 1;
                                    double dx, dy, cnst, G = 6.674e-11;
                                    a[2*(i-1)] = 0;
                                    a[2*i-1] = 0;
                                    
                                    for(int j = 0; j < num_obj; ++j){
                                        if(i != j){
                                            dx = y[2*j] - y[2*i];
                                            dy = y[2*j+1] - y[2*i+1];
                                            cnst = G * mass[j] / ( pow(sqrt(dx*dx + dy*dy), 3) );
                                            a[2*(i-1)] += cnst * dx;
                                            a[2*i-1] += cnst * dy;
                                        }
                                    }
                                }
                                ''').build()
    
    program_fill_v = cl.Program(context, '''
                                __kernel void fill_v(__global const double *v_prev, __global double *v_new, 
                                                     __global const double *a_prev, __global double *a_new,
                                                     const int dt)
                                {
                                    int i = get_global_id(0);
                                    
                                    v_new[i*2] = v_prev[i*2] + (a_new[i*2] + a_prev[i*2+0])*dt/2;
                                    v_new[i*2+1] = v_prev[i*2+1] + (a_new[i*2+1] + a_prev[i*2+1])*dt/2;
                                }''').build()
    
    num_obj = len(y_0)
    y = np.zeros((len(interval), num_obj, 6)) # y = [[x, y, v_x, v_y, a_x, a_y], ...]
    y[0, :, :4] = y_0
    
    temp = np.zeros((2*(num_obj - 1), ))
    mass_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(mass))
    y_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[0, :, 0:2].flatten())
    a_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temp.nbytes)
    program_fill_a.fill_a(queue, (num_obj - 1, ), None, np.int32(num_obj), mass_buffer, y_buffer, a_buffer)
    cl.enqueue_copy(queue, temp, a_buffer)
    y[0, 1:, 4:] = temp.reshape(num_obj - 1, 2)
    
    for t in range(1, len(interval)):
        #print(t, 'begin ', y)
        #x
        temp = np.zeros((2*(num_obj - 1), ))
        y_prev_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[t-1, 1:, :].flatten())
        y_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temp.nbytes)
        program_fill_x.fill_x(queue, (num_obj - 1, ), None, y_prev_buffer, y_buffer, np.int32(dt), np.int32(6))
        cl.enqueue_copy(queue, temp, y_buffer)
        y[t, 1:, :2] = temp.reshape(num_obj - 1, 2)
        #print(t, 'after x ', y)
        
        #a
        temp = np.zeros((2*(num_obj - 1), ))
        #print(temp, y[t, :, 0:2])
        mass_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(mass))
        y_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[t, :, 0:2].flatten())
        a_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temp.nbytes)
        program_fill_a.fill_a(queue, (num_obj - 1, ), None, np.int32(num_obj), mass_buffer, y_buffer, a_buffer)
        cl.enqueue_copy(queue, temp, a_buffer)
        #print(y[t], temp)
        y[t, 1:, 4:] = temp.reshape(num_obj - 1, 2)
        #print(t, 'after a ', y)
        
        #v
        temp = np.zeros((2*(num_obj - 1), ))
        v_prev_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[t-1, 1:, 2:4].flatten())
        a_prev_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[t-1, 1:, 4:].flatten())
        a_new_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y[t, 1:, 4:].flatten())
        v_new_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temp.nbytes)
        program_fill_v.fill_v(queue, (num_obj - 1, ), None, v_prev_buffer, v_new_buffer, a_prev_buffer, a_new_buffer, np.int32(dt))
        cl.enqueue_copy(queue, temp, v_new_buffer)
        y[t, 1:, 2:4] = temp.reshape(num_obj - 1, 2)
        
        #print(t, 'end ', y)
    return y[:, :, :]