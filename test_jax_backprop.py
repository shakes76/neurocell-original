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
lr = 0.01

#===========
##data
X = np.arange(-30, 30, 0.5, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -10) & (X[:,0] < 10) ] = 1.0

plt.plot(X, Y)
plt.show()

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

nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
# nethood = filters.forwardPropagation(dtype=jnp.float32, norm=True) # 8 connected kernel

universe = jax.lax.dynamic_update_slice(universe, X[0].reshape(-1,1), input_offset)
print(X[0], Y[0])

#compute convolutions to forward propagate
#convolution is the time reversed correlation with the kernel.
#hence convolution with forward is equivalent to correlation with backward
result = jsignal.convolve2d(universe, nethood_forward, mode='same')
for i in range(depth-1):
    result = jsignal.convolve2d(result, nethood_forward, mode='same')

output = jax.lax.dynamic_slice(result, output_offset, output_size)
print(output)

#compute loss
cost = Y[0].reshape(-1,1)-output
step = -lr*cost #take negative step
print("Y:", Y[0], "cost:", cost, "step:", step)

#write step and backprop to get updates to weights required
backprop = jax.lax.dynamic_update_slice(result, step, output_offset)

# backprop = jsignal.convolve2d(backprop, nethood_backward, mode='same')
for i in range(depth):
    backprop = jsignal.convolve2d(backprop, nethood_backward, mode='same')

#apply corrections
universe_update = universe + backprop
#restore input (data consistency)
universe_update = jax.lax.dynamic_update_slice(universe_update, X[0].reshape(-1,1), input_offset)

#forward prop
result2 = jsignal.convolve2d(universe_update, nethood_forward, mode='same')
for i in range(depth-1):
    result2 = jsignal.convolve2d(result2, nethood_forward, mode='same')

output2 = jax.lax.dynamic_slice(result2, output_offset, output_size)
print(output2)

#recon input
# recon = jax.lax.dynamic_slice(backprop, input_offset, input_size)
# print(recon)

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
