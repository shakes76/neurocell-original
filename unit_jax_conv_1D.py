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
print("Input Coords:", input_offset)

#===========
# nethood = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
print("nethood:", nethood.shape, nethood)

universe = jax.lax.dynamic_update_slice(universe, inputs, input_offset)

# compute convolutions and plot
# convolution is the time reversed correlation with the kernel.
# hence convolution with forward is equivalent to correlation with backward
result1 = jsignal.convolve2d(universe, nethood, mode='same')

#===========
# compute the 1D FFT version of conv 
# place filter in same space
nethood = jnp.fliplr(nethood) #flip to make it conv as oppose to correlation
width = nethood.shape[0]
print("width:", width)

# padding = (N-width)//2
# odd_pad = 0
# if (N-width)%2 == 1: #odd
#     odd_pad = 1

# filter = jnp.pad(nethood, ((padding,padding+odd_pad), (0,0)), mode='constant')
filter = jnp.pad(nethood, ((0,(N-width)), (0,0)), mode='constant')
# filter = jnp.pad(nethood, (((N-width),0), (0,0)), mode='constant')
print("filter:", filter.shape, filter)

# FFTs of filter
filter_conv1D = jnp.fft.fft(filter, axis=0) #ffts of rows

# FFTs of universe
universe_conv1D = jnp.fft.fft(universe, axis=0) #ffts of rows

#Conv via CCP
# print(universe_conv1D[:,input_left-1:input_left+width-1])
result_conv1D = filter_conv1D * universe_conv1D[:,input_left:input_left+width]

#inverse FFT
result_block = jnp.fft.ifft(result_conv1D, axis=0) #ffts of rows
result_block = jnp.real(result_block)
# print("result_block:", result_block.shape, result_block)

plt.imshow(result_block)
plt.title("Result Block")
plt.show()

#copy back
result2 = jax.lax.dynamic_update_slice(universe, result_block, (0,input_left))

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result1)
plt.title("Result 1")
plt.show()

plt.imshow(result2)
plt.title("Result 2")
plt.show()
