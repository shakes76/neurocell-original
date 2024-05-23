'''
Unit test for effect of filters
'''
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import matplotlib.pyplot as plt

import filters #local

#parameters
N = 7
samples = 120
length = 1
num_classes = 1
lr = 0.1

#===========
##data
X = np.arange(-30, 30, 0.5, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -10) & (X[:,0] < 10) ] = 1.0

plt.plot(X, Y)
plt.show()

#helpers
def forward_propagate(cells):
    cells = jsignal.convolve2d(cells, nethood_forward, mode='same')
    cells = jax.nn.standardize(cells, axis=(0,1))
    return jax.nn.sigmoid(cells)

def backward_propagate(cells):
    cells = jsignal.convolve2d(cells, nethood_backward, mode='same')
    cells = jax.nn.standardize(cells, axis=(0,1))
    return jax.nn.sigmoid(cells)

def clear_cells(universe, offset, after=False):
    if after:
        result = universe.at[:,offset[1]+1:].set(0)  #clear cells after
    else:
        result = universe.at[:,:offset[1]].set(0) #clear cells before
    return result

#===========
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
universe = jnp.zeros((N,N))
key = jax.random.key(42)
depth = output_offset[1]-input_offset[1]
print("depth:", depth)

#random init
new_key, subkey = jax.random.split(key)
noise = 1.0 * jax.random.normal(subkey, universe.shape)

#random universe
universe += noise

#filters for spatial derivatives
nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
# nethood = filters.forwardPropagation(dtype=jnp.float32, norm=True) # 8 connected kernel

universe = jax.lax.dynamic_update_slice(universe, X[0].reshape(-1,1), input_offset)
# universe = universe.at[:,:input_offset[1]].set(0) #clear cells before
universe = clear_cells(universe, input_offset, after=False)
universe = jax.lax.dynamic_update_slice(universe, Y[0].reshape(-1,1), output_offset)
# universe = universe.at[:,output_offset[1]+1:].set(0)  #clear cells after
universe = clear_cells(universe, output_offset, after=True)
print(X[0], Y[0])

#compute convolutions to forward propagate
#convolution is the time reversed correlation with the kernel.
#hence convolution with forward is equivalent to correlation with backward
result = forward_propagate(universe)
for i in range(depth-1):
    result = forward_propagate(result)

#get output
output = jax.lax.dynamic_slice(result, output_offset, output_size)
print(output)

#compute loss
cost = Y[0].reshape(-1,1)-output
step = lr*cost #take negative step
print("Y:", Y[0], "cost:", cost, "step:", step)

#write step and backprop to get updates to weights required
backprop = jax.lax.dynamic_update_slice(result, step, output_offset)
backprop = clear_cells(backprop, output_offset, after=True)

for i in range(depth):
    backprop = backward_propagate(backprop)

#apply corrections
universe_update = universe - backprop
#restore input (data consistency)
universe_update = jax.lax.dynamic_update_slice(universe_update, X[0].reshape(-1,1), input_offset)

#forward prop
result2 = forward_propagate(universe_update)
for i in range(depth-1):
    result2 = forward_propagate(result2)

output2 = jax.lax.dynamic_slice(result2, output_offset, output_size)
print(output2)

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result)
plt.title("Result")
plt.show()

plt.imshow(backprop)
plt.title("Backprop")
plt.show()

plt.imshow(universe_update)
plt.title("Updated")
plt.show()

plt.imshow(result2)
plt.title("Result2")
plt.show()
