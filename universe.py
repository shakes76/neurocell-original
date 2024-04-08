'''
Universe object for Von Neumann networks
'''
import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.scipy.signal as jsignal
import haiku as hk

import filters #local

class Universe:
    '''
    Represents cellular structured universe of Von Neumann networks
    '''
    def __init__(self, N):
        self.N = N
        self.grid_size = (N,N)
        self.key = jax.random.key(42)
        #filters for spatial derivatives
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.zero_init()

    def zero_init(self):
        self.universe = jnp.zeros(self.grid_size)

    def random_init(self):
        new_key, subkey = jax.random.split(self.key)
        self.new_key = new_key
        self.universe = self.universe + 1.0 * jax.random.normal(subkey, self.grid_size)

    def get_cells(self):
        return self.universe
    
    def clear_cells_vertical(self, universe, offset, after=False):
        if after:
            result = universe.at[:,offset[1]+1:].set(0)  #clear cells after
        else:
            result = universe.at[:,:offset[1]].set(0) #clear cells before
        return result

    def set_input_vector(self, inputs, offset=None):
        '''
        Set in vector at address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        cells = jax.lax.dynamic_update_slice(self.universe, inputs, self.in_offset)
        self.universe = self.clear_cells_vertical(cells, self.in_offset, after=False)
    
    def set_output_vector(self, outputs, offset=None):
        '''
        Write out vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        cells = jax.lax.dynamic_update_slice(self.universe, outputs, self.out_offset)
        self.universe = self.clear_cells_vertical(cells, self.out_offset, after=True)
    
    def get_input_vector(self, input_size, offset=None):
        '''
        Read in vector from address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.in_offset, input_size)
    
    def get_output_vector(self, output_size, offset=None):
        '''
        Read out vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.out_offset, output_size)
    
    def forward_propagate(self, activate=nn.sigmoid):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(self.universe, self.nethood_forward, mode='same')
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        self.universe = activate(cells)

    def backward_propagate(self, activate=nn.sigmoid):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(self.universe, self.nethood_backward, mode='same')
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        self.universe = activate(cells)

class UniversePerceptron(hk.Module):
    '''
    Represents cellular structured universe of Von Neumann networks
    '''
    def __init__(self, N, name=None):
        super().__init__(name=name)
        self.N = N
        self.grid_size = (N,N)
        #filters for spatial derivatives
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.init()

    def init(self):
        # w_init = jnp.zeros
        w_init = hk.initializers.RandomNormal()
        self.universe = hk.get_parameter("w", shape=self.grid_size, dtype=jnp.float32, init=w_init)

    def get_cells(self):
        return self.universe
    
    def clear_cells_vertical(self, universe, offset, after=False):
        if after:
            result = universe.at[:,offset[1]+1:].set(0)  #clear cells after
        else:
            result = universe.at[:,:offset[1]].set(0) #clear cells before
        return result

    def set_input_vector(self, inputs, offset=None):
        '''
        Set in vector at address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        self.universe = jax.lax.dynamic_update_slice(self.universe, inputs, self.in_offset)
        return self.clear_cells_vertical(self.universe, self.in_offset, after=False)
    
    def set_output_vector(self, outputs, offset=None):
        '''
        Write out vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        self.universe = jax.lax.dynamic_update_slice(self.universe, outputs, self.out_offset)
        return self.clear_cells_vertical(self.universe, self.out_offset, after=True)
    
    def get_input_vector(self, input_size, offset=None):
        '''
        Read in vector from address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.in_offset, input_size)
    
    def get_output_vector(self, output_size, offset=None):
        '''
        Read out vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.out_offset, output_size)
    
    def forward_propagate(self, cells, activate=nn.sigmoid):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)

    def backward_propagate(self, cells, activate=nn.sigmoid):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_backward, mode='same')
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)
    
    def __call__(self, x, in_offset, output_size, out_offset, activate=nn.sigmoid):
        #input
        net = self.set_input_vector(x, in_offset)
        #forward pass
        self.universe = self.forward_propagate(net, activate)
        #read output
        return self.get_output_vector(output_size, out_offset)
