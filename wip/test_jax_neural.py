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
parser.add_argument('--N', '--size', default=128, type=int,
                    metavar='Size', help='Initial grid size of the universe', dest='N')
parser.add_argument('--int', '--interval', default=20, type=int,
                    metavar='Interval', help='Interval between frames in ms', dest='interval')
parser.add_argument('--ell', '--lambda', default=8, type=float,
                    metavar='Lambda', help='Lower bound', dest='ell')
parser.add_argument('--gee', '--gamma', default=16, type=float,
                    metavar='Gamma', help='Upper bound', dest='gee')
parser.add_argument('--pat', '--pattern', default="patterns/pattern.rle", type=str,
                    metavar='Pattern', help='Pattern to insert into the grid', dest='pat')
parser.add_argument('--magma', dest='magma', action='store_const', const=True, default=False,
                    help='Change to the Magma colour map')

args = parser.parse_args()

utils.jax_status()

N = args.N
print("Universe Size (N):", N)
grid_size = (N,N) 
insert_point = (N//4,N//4)
universe = np.zeros(grid_size)
with open(args.pat) as rleFile:
    patterns.insertFromRLE(grid=universe, rleString=rleFile.read(), index=insert_point)
universe = universe.astype(np.float32)
print("Universe Shape:", universe.shape, universe.dtype)

#plot cells
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

plt.gray()
if args.magma:
    plt.magma()

img = plt.imshow(universe, animated=True)

def animate(i):
    """perform animation step"""
    global universe

    #universe = evolve.evolve(universe)
    #universe = evolve.evolve_fast(universe) 
    universe = evolve.evolve_neural(universe, Lambda=args.ell, Gamma=args.gee) 
    #~ universe = evolve.evolve_neural_nz(universe, Lambda=args.ell, Gamma=args.gee) 
    #~ universe = evolve.evolve_network(universe, Lambda=args.ell, Gamma=args.gee) 
    print("i", i, "universe count", jnp.sum(universe))

    img.set_array(universe)
    
    return img,

interval = args.interval #ms

#animate 24 frames with interval between them calling animate function at each frame
ani = animation.FuncAnimation(fig, animate, frames=2048, interval=interval, blit=True)

plt.show()
