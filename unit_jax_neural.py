'''
Implementation of the Neural Cell in JAX single iteration
'''
import utils #local
import patterns #local
import evolve
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Neurocell JAX')
parser.add_argument('--ell', '--lambda', default=4, type=float,
                    metavar='Lambda', help='Lower bound', dest='ell')
parser.add_argument('--gee', '--Gamma', default=8, type=float,
                    metavar='Gamma', help='Upper bound', dest='gee')

args = parser.parse_args()

utils.jax_status()

N = 32
insert_point = (16,16)
grid_size = (N,N) 
universe = np.zeros(grid_size) 

#universe = patterns.insertGlider(universe, index=insert_point)
# universe = patterns.insertGliderGun(universe, index=insert_point)
universe = patterns.insertInputVertical(universe, input=np.ones(5), index=insert_point)
universe = universe.astype(np.float32)
print("Universe Shape:", universe.shape, universe.dtype)

#plot cells
import matplotlib.pyplot as plt

fig = plt.figure()

plt.gray()

img = plt.imshow(universe, animated=True)

plt.show()

# universe = evolve.evolve_neural(universe, Lambda=args.ell, Gamma=args.gee) 
universe = evolve.evolve_network(universe, Lambda=args.ell, Gamma=args.gee) 
print("Universe Shape (updated):", universe.shape, universe.dtype)

img = plt.imshow(universe, animated=True)

plt.show()

# universe = evolve.evolve_neural(universe, Lambda=args.ell, Gamma=args.gee) 
universe = evolve.evolve_network(universe, Lambda=args.ell, Gamma=args.gee) 
print("Universe Shape (updated 2):", universe.shape, universe.dtype)

img = plt.imshow(universe, animated=True)

plt.show()

