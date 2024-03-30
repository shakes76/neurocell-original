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
length = 4

#===========
input_size = (length,1)
input_offset = (2,4)
inputs = jnp.ones(input_size)
universe = jnp.zeros((N,N))
key = jax.random.key(42)

#random init
new_key, subkey = jax.random.split(key)
noise = 0.1 * jax.random.normal(subkey, universe.shape)

#random universe
universe += noise

# nethood = filters.vonNeumann(dtype=jnp.float32, norm=True) # 8 connected kernel
# nethood = filters.backward(dtype=jnp.float32, norm=True) # 8 connected kernel
nethood = filters.forward(dtype=jnp.float32, norm=True) # 8 connected kernel
# nethood = filters.forwardPropagation(dtype=jnp.float32, norm=True) # 8 connected kernel

universe = jax.lax.dynamic_update_slice(universe, inputs, input_offset)

#compute convolutions and plot
#convolution is the time reversed correlation with the kernel.
#hence convolution with forward is equivalent to correlation with backward
result1 = jsignal.convolve2d(universe, nethood, mode='same')

result2 = jsignal.convolve2d(result1, nethood, mode='same')

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result1)
plt.title("Result 1")
plt.show()

plt.imshow(result2)
plt.title("Result 2")
plt.show()
