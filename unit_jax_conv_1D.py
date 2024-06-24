'''
Unit test for effect of filters
'''
import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import matplotlib.pyplot as plt

import filters #local

#parameters
N = 8
depth = 7
length = 4

#===========
center = N//2
input_top = center-length//2
input_left = center-1
input_size = (length,1)
input_offset = (input_top,input_left)
inputs = jnp.ones(input_size)
universe = jnp.zeros((N,depth))

# nethood = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
print(nethood)

universe = jax.lax.dynamic_update_slice(universe, inputs, input_offset)

#compute convolutions and plot
#convolution is the time reversed correlation with the kernel.
#hence convolution with forward is equivalent to correlation with backward
result1 = jsignal.convolve2d(universe, nethood, mode='same')

# result2 = jsignal.convolve2d(result1, nethood, mode='same')

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result1)
plt.title("Result 1")
plt.show()

# plt.imshow(result2)
# plt.title("Result 2")
# plt.show()
