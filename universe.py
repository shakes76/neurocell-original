'''
Universe object for Von Neumann networks
'''
import jax
import jax.numpy as jnp
import haiku as hk

class Universe(hk.Module):
    '''
    Represents cellular structured universe of Von Neumann networks
    '''
    def __init__(self, N):
        self.N = N
        self.grid_size = (N,N) 

    def init(self):
        # self.universe = jnp.zeros(self.grid_size)
        w_init = hk.initializers.RandomNormal()
        self.universe = hk.get_parameter("universe", shape=self.grid_size, dtype=jnp.float32, init=w_init)

    def set_input_vector(self, inputs, offset=None):
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        return jax.lax.dynamic_update_slice(self.universe, inputs, self.in_offset)
    
    def get_output_vector(self, outputs, offset=None):
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        input_size = outputs.shape
        return jax.lax.dynamic_slice(self.universe, self.out_offset, input_size)

