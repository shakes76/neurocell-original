'''
Implementation of the Neural Cell in JAX
'''
import utils #local
import patterns #local
import evolve
import numpy as np
import jax.numpy as jnp

import argparse

parser = argparse.ArgumentParser(description='Neurocell JAX')
parser.add_argument('--int', '--interval', default=20, type=int,
                    metavar='Interval', help='Interval between frames in ms', dest='interval')
parser.add_argument('--ell', '--lambda', default=8, type=float,
                    metavar='Lambda', help='Lower bound', dest='ell')
parser.add_argument('--gee', '--Gamma', default=16, type=float,
                    metavar='Gamma', help='Upper bound', dest='gee')

args = parser.parse_args()

utils.jax_status()

N = 128
insert_point = (16,16)
grid_size = (N,N) 
universe = np.zeros(grid_size) 

#universe = patterns.insertGlider(universe, index=insert_point)
universe = patterns.insertGliderGun(universe, index=insert_point)
universe = universe.astype(np.float32)
print("Universe Shape:", universe.shape, universe.dtype)

#universe = evolve.evolve_neural(universe) 
#print("Universe Shape (updated):", universe.shape, universe.dtype)

#plot cells
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

plt.gray()

img = plt.imshow(universe, animated=True)

def animate(i):
    """perform animation step"""
    global universe

    #universe = evolve.evolve(universe)
    #universe = evolve.evolve_fast(universe) 
    universe = evolve.evolve_neural(universe, Lambda=args.ell, Gamma=args.gee) 
    print("i", i, "universe count", jnp.sum(universe))

    img.set_array(universe)
    
    return img,

interval = args.interval #ms

#animate 24 frames with interval between them calling animate function at each frame
ani = animation.FuncAnimation(fig, animate, frames=2048, interval=interval, blit=True)

plt.show()

#universe = evolve.evolve_neural(universe) 
#print("Universe Shape (updated):", universe.shape, universe.dtype)

#img = plt.imshow(universe, animated=True)

#plt.show()

