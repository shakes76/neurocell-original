'''
Universe object for Von Neumann networks
'''
import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.scipy.signal as jsignal
import haiku as hk

import filters #local

class Universe(hk.Module):
    '''
    Represents cellular structured universe of Von Neumann networks
    '''
    def __init__(self, N, name=None):
        super().__init__(name=name)
        self.N = N
        self.grid_size = (N,N)
        self.nethood = filters.forward(dtype=jnp.float32, norm=True) # 8 connected kernel
        # self.nethood = filters.forwardPropagation(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood)
        self.init()

    def init(self):
        # w_init = jnp.zeros
        w_init = hk.initializers.RandomNormal()
        self.universe = hk.get_parameter("w", shape=self.grid_size, dtype=jnp.float32, init=w_init)

    def get_cells(self):
        return self.universe

    def set_input_vector(self, inputs, offset=None):
        '''
        Set vector at address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        return jax.lax.dynamic_update_slice(self.universe, inputs, self.in_offset)
    
    def get_output_vector(self, output_size, offset=None):
        '''
        Read vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.out_offset, output_size)
    
    def __call__(self, x, in_offset, output_size, out_offset, act=nn.relu):
        #input
        net = self.set_input_vector(x, in_offset)
        #forward pass
        #get weighted sum of neighbors using convolution
        net = jsignal.convolve2d(net, self.nethood, mode='same')
        #stats, ignoring zeros
        # avg = jnp.nanmean(jnp.where(x!=0, x, jnp.nan))
        # var = jnp.nanvar(jnp.where(x!=0, x, jnp.nan))
        #norm
        net = nn.standardize(net, axis=(0,1))
        #excite
        self.universe = act(net)
        #read output
        return self.get_output_vector(output_size, out_offset)
