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
# input_top = 0
input_top = center-length//2
input_left = center-1
input_size = (length,1)
input_offset = (input_top,input_left)
inputs = jnp.ones(input_size)
universe = jnp.zeros((N,depth))
print("Input Coords:", input_offset)

#===========
# nethood = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
# nethood = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
nethood = filters.smooth(dtype=jnp.float32, norm=False) # 8 connected kernel
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
width = nethood.shape[1]
print("width:", width)

# filter = jnp.pad(nethood, ((padding,padding+odd_pad), (0,0)), mode='constant')
filter = jnp.pad(nethood, ((0,(N-width)), (0,0)), mode='constant')
# filter = jnp.pad(nethood, (((N-width),0), (0,0)), mode='constant')
filter = jnp.roll(filter, -1, axis=0) #make sure centre pixel is first in filter by periodic shift
print("filter:", filter.shape, filter)

# FFTs of filter
filter_fft = jnp.fft.fft(filter, axis=0) #ffts of rows

#next column to process for forward pass
index = input_left+1

#sub region around index to process
sub_region = universe[:,index-1:index-1+width]

# FFTs of universe
sub_region_fft = jnp.fft.fft(sub_region, axis=0) #ffts of rows
# print("FFTs:", sub_region_fft.shape, sub_region_fft)

#Conv via CCP
result_conv1D = filter_fft * sub_region_fft
result_conv1D = jnp.sum(result_conv1D, axis=-1).reshape(-1,1) #reshape promotes to column vec
# print(result_conv1D.shape, result_conv1D)

#inverse FFT
result_block = jnp.fft.ifft(result_conv1D, axis=0) #ffts of rows
result_block = jnp.real(result_block)
# print("result_block:", result_block.shape, result_block)

plt.imshow(result_block)
plt.title("Result Block")
plt.show()

#copy back
result2 = jax.lax.dynamic_update_slice(universe, result_block, (0,index))

plt.imshow(universe)
plt.title("Initial")
plt.show()

plt.imshow(result1)
plt.title("Result 1")
plt.show()

plt.imshow(result2)
plt.title("Result 2")
plt.show()
