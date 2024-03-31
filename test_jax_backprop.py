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

#random init
new_key, subkey = jax.random.split(key)
noise = 0.1 * jax.random.normal(subkey, universe.shape)

#random universe
# universe += noise

nethood_backward = filters.backward(dtype=jnp.float32, norm=True) # 8 connected kernel
nethood_forward = filters.forward(dtype=jnp.float32, norm=True) # 8 connected kernel
# nethood = filters.forwardPropagation(dtype=jnp.float32, norm=True) # 8 connected kernel

universe = jax.lax.dynamic_update_slice(universe, X[0].reshape(-1,1), input_offset)
print(X[0], Y[0])

#compute convolutions and plot
#convolution is the time reversed correlation with the kernel.
#hence convolution with forward is equivalent to correlation with backward
result1 = jsignal.convolve2d(universe, nethood_forward, mode='same')
result2 = jsignal.convolve2d(result1, nethood_forward, mode='same')
result3 = jsignal.convolve2d(result2, nethood_forward, mode='same')
result4 = jsignal.convolve2d(result3, nethood_forward, mode='same')

output = jax.lax.dynamic_slice(result4, output_offset, output_size)
print(output)

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result1)
plt.title("Result 1")
plt.show()

plt.imshow(result2)
plt.title("Result 2")
plt.show()

plt.imshow(result3)
plt.title("Result 3")
plt.show()

plt.imshow(result4)
plt.title("Result 4")
plt.show()
