'''
Unit test of forward propagation of neurocells
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import universe #local
import filters #local

#parameters
N = 7
length = 1
depth = 7
num_classes = 1

#===========
center = N//2
# input_top = 0
input_top = center-length//2
input_left = center-1
input_size = (length,1)
input_offset = (input_top,input_left)
activation = jax.nn.tanh
inputs = jnp.ones(input_size)
weights = jnp.ones((N,depth))
states = jnp.ones((N,depth))
biases = jnp.zeros((N,depth))
cells = jnp.zeros((N,depth))
print("Input Coords:", input_offset)

#===========
# nethood = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
# nethood = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood = filters.smooth(dtype=jnp.float32, norm=False) # 8 connected kernel
print("nethood:", nethood.shape, nethood)

cells = jax.lax.dynamic_update_slice(cells, inputs, input_offset)

plt.imshow(cells)
plt.title("Cells")
plt.show()

index = input_left+1

result_block = universe._forward_fast(index, cells, weights, biases, states, nethood, activation, False)
print("Result Block:", result_block.shape, result_block)

cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index))

plt.imshow(cells)
plt.title("Forward")
plt.show()

# plt.imshow(weights)
# plt.title("Weights")
# plt.show()
