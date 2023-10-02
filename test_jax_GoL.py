'''
Implementation of the Game of Life (GoL) in JAX
'''
import utils #local
import patterns
#import jax.numpy as jnp
import numpy as np

utils.jax_status()

N = 64
insert_point = (32,32)
grid_size = (N,N) 
universe = np.zeros(grid_size) 

universe = patterns.insertGlider(universe, index=insert_point)

#plot cells
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

plt.gray()

img = plt.imshow(universe, animated=True)

plt.show()

