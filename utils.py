'''
Utility functions for library
'''
import jax
import jax.numpy as jnp
import imageio as iio

def jax_status():
    print("Total Devices:",jax.devices())
    print("Used Devices:",jax.local_devices())
    print("Using", jnp.ones(3).device_buffer.device())

def load_sprite(filename):
    '''
    Loads sprite image as cells of different states
    Assumes that inputs are < 100, neurons < 200 and outputs < 255.
    Returns size and the list of corrdinates of the inputs, neurons and outputs
    '''
    model_sprite = iio.imread(filename)
    N = max(model_sprite.shape[0], model_sprite.shape[1])
    print("N:", N)

    inputs = []
    neurons = []   
    outputs = []
    for i, row in enumerate(model_sprite):
        for j, col in enumerate(row):
            point = (i,j)
            if col == 0:
                continue
            elif col < 100:
                inputs.append(point)
            elif col < 200:
                neurons.append(point)
            else: #col < 255
                outputs.append(point)
    
    return N, inputs, neurons, outputs

