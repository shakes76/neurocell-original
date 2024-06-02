'''
Universe object for Von Neumann networks
'''
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.scipy.signal as jsignal
import haiku as hk

import filters #local

#kernels for larger networks
kernel_backward =  [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]
kernel_forward =   [[0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]]

class Universe:
    '''
    Represents cellular structured universe of Von Neumann networks
    '''
    def __init__(self, N, dtype=jnp.float32):
        self.epsilon = 1e-6
        self.N = N
        self.grid_size = (N,N)
        self.dtype = dtype
        self.states_dtype = jnp.uint8
        self.key = jax.random.key(42)
        #filters for spatial derivatives
        #self.nethood_backward = jnp.array(kernel_backward, dtype=dtype) # 8 connected kernel
        #self.nethood_forward = jnp.array(kernel_forward, dtype=dtype) # 8 connected kernel
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.initialize() #universe and prev_universe

    def initialize(self):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.states = jnp.zeros(self.grid_size, dtype=self.states_dtype)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size, dtype=self.dtype)
        self.key = new_key

        self.save_state()
        
    def initialize_vertical(self, in_size, out_size, in_offset=None, out_offset=None):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.states = jnp.ones(self.grid_size, dtype=self.states_dtype)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size)
        self.biases = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.key = new_key

        #setup in/out weights
        self.in_offset = (self.N//4, self.N//8)
        if in_offset:
            self.in_offset = in_offset
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if out_offset:
            self.out_offset = out_offset
        in_passives = jnp.full(in_size, 2, dtype=self.states_dtype)
        out_passives = jnp.full(in_size, 2, dtype=self.states_dtype)
        inputs = jnp.ones(in_size, dtype=self.weights.dtype)
        outputs = jnp.ones(out_size, dtype=self.weights.dtype)
        weights = self.clear_cells_vertical(self.weights, self.in_offset, after=False)
        self.weights = jax.lax.dynamic_update_slice(weights, inputs, self.in_offset)
        states = self.clear_cells_vertical(self.states, self.in_offset, after=False)
        self.states = jax.lax.dynamic_update_slice(states, in_passives, self.in_offset)
        weights = self.clear_cells_vertical(self.weights, self.out_offset, after=True)
        self.weights = jax.lax.dynamic_update_slice(weights, outputs, self.out_offset)
        states = self.clear_cells_vertical(self.states, self.out_offset, after=True)
        self.states = jax.lax.dynamic_update_slice(states, out_passives, self.out_offset)

        self.save_state()

    def get_cells(self):
        return self.universe
        
    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.biases
    
    def get_states(self):
        return self.states
    
    def save_state(self):
        self.previous_universe = self.universe
        
    def restore_state(self):
        self.universe = self.previous_universe
        
    def reset_state(self):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
    
    def clear_cells_vertical(self, universe, offset, after=False):
        '''
        Clears cells upto and including offset positions
        '''
        if after:
            result = universe.at[:,offset[1]:].set(0)  #clear cells after
        else:
            result = universe.at[:,:offset[1]+1].set(0) #clear cells before
        return result

    def set_input_vector(self, inputs, offset=None):
        '''
        Set in vector at address with size
        '''
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        cells = self.clear_cells_vertical(self.universe, self.in_offset, after=False)
        self.universe = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        # weights = self.clear_cells_vertical(self.weights, self.in_offset, after=False)
        # self.weights = jax.lax.dynamic_update_slice(weights, inputs, self.in_offset)
    
    def set_output_vector(self, outputs, offset=None):
        '''
        Write out vector from address with size
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        cells = self.clear_cells_vertical(self.universe, self.out_offset, after=True)
        self.universe = jax.lax.dynamic_update_slice(cells, outputs, self.out_offset)
        # weights = self.clear_cells_vertical(self.weights, self.out_offset, after=True)
        # self.weights = jax.lax.dynamic_update_slice(weights, outputs, self.out_offset)

    def set_layer_vertical(self, layer_size, offset=None):
        passives = jnp.full(layer_size, 2, dtype=self.states_dtype)
        self.states = jax.lax.dynamic_update_slice(self.states, passives, offset)
    
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

    def shift(self, value):
        self.universe = self.universe - value #account for sigmoid etc.
    
    def forward(self, cells, weights, biases, activate=nn.sigmoid, norm=True):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #check states
        #if passive apply activation, else signal state so weight
        cells = jnp.where(self.states==1, cells*weights, cells) #weight if signal
        cells = jnp.where(jnp.logical_and(self.states==2,cells!=0), activate(cells+biases), cells) #activate if passive
        #apply weights
        # cells = cells * self.weights
        # if norm:
        #     #norm
        #     cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        # return activate(cells)
        return cells
    
    def forward_propagate(self, activate=nn.sigmoid, norm=True):
        self.universe = self.forward(self.universe, self.weights, self.biases, activate=activate, norm=norm)

    def backward(self, cells, activate=nn.sigmoid, norm=True):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_backward, mode='same')
        #apply weights
        cells = cells * self.weights
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)
        '''
        #run the network in reverse
        #excite
        cells = activate(cells)
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #apply weights
        cells = cells / (self.weights+1e-8)
        #get weighted sum of neighbors using convolution
        return jsignal.convolve2d(cells, self.nethood_backward, mode='same')'''
    
    def backward_propagate(self, activate=nn.sigmoid, norm=True):
        #excite and return
        self.universe = self.backward(self.universe, activate=activate, norm=norm)

    def loss(self, cells, depth, labels, inputs, in_offset, output_size, out_offset, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        '''
        Cross entropy 
        '''
        #set input
        # cells = self.clear_cells_vertical(cells, in_offset, after=False)
        cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)
        #forward
        for i in range(1, depth):
            cells = self.forward(cells, activate=activate, norm=norm)
        cells = self.forward(cells, activate=final_activate, norm=norm)
        #read output
        logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
        return -1 * jnp.mean(labels * jnp.log(logits + self.epsilon) + (1. - labels) * jnp.log(1. - logits + self.epsilon)) #CE
        # return jnp.mean((labels - logits)**2) #MSE
        
    def loss_perturb(self, params, deltas, labels, inputs, in_offset, output_size, out_offset, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        '''
        Cross entropy 
        We use the perturbation trick, so get the layer by layer grad values for the overall loss
        See https://github.com/google/jax/discussions/5336#discussioncomment-269983 
        '''
        depth = len(deltas)
        weights, biases = params
        #set input
        # cells = self.clear_cells_vertical(cells, in_offset, after=False)
        cells = jax.lax.dynamic_update_slice(self.universe, inputs, in_offset)
        #forward
        for i in range(1, depth):
            cells = self.forward(cells, weights, biases, activate=activate, norm=norm)
            # cells = cells + deltas[i-1] #have grad return layer grad
        cells = self.forward(cells, weights, biases, activate=final_activate, norm=norm)
        # cells = cells + deltas[depth-1]
        # print("activations size:", len(activations))

        #read output
        logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
        loss_value = -1 * jnp.mean(labels * jnp.log(logits + self.epsilon) + (1. - labels) * jnp.log(1. - logits + self.epsilon)) #CE

        return loss_value

    def value_and_grad(self, depth, labels, inputs, in_offset, out_size, out_offset):
        return jax.value_and_grad(self.loss)(self.universe, depth, labels, inputs, in_offset, out_size, out_offset, norm=False)
        
    def value_and_grad_perturb(self, depth, labels, inputs, in_offset, out_size, out_offset):
        '''
        We use delta values to perturb the grad of the loss function
        Setting delta to zero means the grad will return the gradient of that layer
        '''
        deltas = []
        for i in range(1,depth+1):
            deltas.append(0.0)
        deltas = tuple(deltas)
        # print("deltas:", deltas)
        params = [self.weights, self.biases]
        return jax.value_and_grad(self.loss_perturb)(params, deltas, labels, inputs, in_offset, out_size, out_offset, norm=False)

    def value_and_grad_update(self, lr, depth, labels, inputs, in_offset, out_size, out_offset):
        self.loss_value, self.grads_output = self.value_and_grad_perturb(depth, labels, inputs, in_offset, out_size, out_offset)
        # print("loss_value", self.loss_value)
        # print("layer grads", len(self.layer_grads), self.layer_grads[0].shape)
        # print("grads", len(self.grads_output), self.grads_output)
        # print("grads", len(self.grads_output[0]), self.grads_output[0])
        # print("grads", len(self.grads_output[1]), self.grads_output[1])

        #compose per layer grads
        self.grads = self.grads_output[0]
        self.bias_grads = self.grads_output[1]
        # self.grads = jnp.zeros_like(self.universe)
        # for i in range(1,depth):
        #     self.grads = self.grads.at[:,self.out_offset[1]-i].set(self.layer_grads[depth-i-1][:,self.out_offset[1]-i])
        # self.grads = self.grads.at[:,self.in_offset[1]].set(self.grads_output[0][:,self.in_offset[1]])

        #apply corrections
        self.weights = self.weights - lr*self.grads
        self.biases = self.biases - lr*self.bias_grads

        return self.loss_value, self.grads, self.bias_grads, self.weights

    def gradient_update(self, lr, step, depth, inputs, input_size, in_offset=None, out_offset=None, activate=nn.sigmoid):
        '''
        Computes the gradient based on the step and backprops it
        Computes and applies the updates to the saved state, assigns to current state
        '''
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if out_offset:
            self.out_offset = out_offset

        #tracking changes
        updates = jnp.zeros_like(self.universe)
        #write step and backprop to get updates to weights required
        # updates = jax.lax.dynamic_update_slice(updates, step, self.out_offset) #include out

        #write step and backprop to get updates to weights required
        cells = self.clear_cells_vertical(self.universe, self.out_offset, after=True)
        cells = jax.lax.dynamic_update_slice(cells, step, self.out_offset)
        # cells = nn.sigmoid(cells)
        #apply weights
        # cells = cells / (self.weights+1e-8)
        updates = updates.at[:,self.out_offset[1]].set(cells[:,self.out_offset[1]])

        cells = self.backward(cells, activate=activate, norm=False)
        updates = updates.at[:,self.out_offset[1]-1].set(cells[:,self.out_offset[1]-1])
        for i in range(2,depth+1): #+1 includes in
            cells = self.backward(cells, activate=activate, norm=False)
            # cells = cells - 0.5 #account for sigmoid etc.
            #track updated weights per 'layer'
            updates = updates.at[:,self.out_offset[1]-i].set(cells[:,self.out_offset[1]-i])
            # self.universe = self.universe + cells
        # print("backprop:", cells)
        #copy updates to universe
        self.universe = cells.at[:,(self.out_offset[1]-depth):(self.out_offset[1]+1)].set(updates[:,(self.out_offset[1]-depth):(self.out_offset[1]+1)])
        
        #rescale by predicted input
        #take lr step
        # self.universe = lr*self.universe
        #rescale changes to be normalised
        # self.universe = jax.nn.standardize(self.universe, axis=(0,1))
        
        #L2 norm of weights
        # jnp.sum(self.universe)
        # self.universe = self.universe/jnp.abs(self.universe)

        #apply corrections
        # scaling = 1.0
        self.weights = self.weights - lr*self.universe
        # self.weights = jax.nn.standardize(self.weights, axis=(0,1))
        #restore input (data consistency)
        # self.universe = jax.lax.dynamic_update_slice(cells, X[0].reshape(-1,1), input_offset)

    def save(self, model_name):
        jnp.savez(model_name, weights=self.weights, biases=self.biases, states=self.states, allow_pickle=False)

    def load(self, filename):
        with jnp.load(filename, allow_pickle=False) as data:
            self.weights = data['weights']
            self.biases = data['biases']
            self.states = data['states']
        self.N = self.weights.shape[0]
        self.grid_size = self.weights.shape
        self.dtype = self.weights.dtype
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)

class VerticalCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting verticval cells."""

    def __init__(self, in_size, out_size, in_offset, out_offset, is_states=False, clear=True):
        """
        Constructs a :class:`RandomNormal` initializer with vertical cells.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.is_states = is_states
        self.clear = clear

    def clear_cells_vertical(self, cells, offset, after=False):
        '''
        Clears cells upto and including offset positions
        '''
        if after:
            result = cells.at[:,offset[1]:].set(0)  #clear cells after
        else:
            result = cells.at[:,:offset[1]+1].set(0) #clear cells before
        return result

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        if self.is_states:
            cells = jnp.ones(shape, dtype=dtype)
            inputs = jnp.full(self.in_size, 2, dtype=dtype)
            outputs = jnp.full(self.in_size, 2, dtype=dtype)
        else:
            # real_dtype = jnp.finfo(dtype).dtype
            # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
            # upper = 3.0
            # lower = -3.0
            # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
            cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
            inputs = jnp.ones(self.in_size, dtype=dtype)
            outputs = jnp.ones(self.out_size, dtype=dtype)

        if self.clear:
            cells = self.clear_cells_vertical(cells, self.in_offset, after=False)
            cells = self.clear_cells_vertical(cells, self.out_offset, after=True)
        
        cells = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        cells = jax.lax.dynamic_update_slice(cells, outputs, self.out_offset)
        return cells

class UniverseMLP(hk.Module):
    '''
    Represents cellular structured universe of Von Neumann networks as a multi-layered perceptron (MLP)
    '''
    def __init__(self, N, in_size, out_size, in_offset, out_offset, layer_size, layer_offset, dtype=jnp.float32, name=None):
        super().__init__(name=name)
        self.N = N
        self.grid_size = (N,N)
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.layer_size = layer_size
        self.layer_offset = layer_offset
        self.depth = self.out_offset[1]-self.in_offset[1]
        self.dtype = dtype
        self.states_dtype = jnp.uint8
        #filters for spatial derivatives
        # self.nethood_backward = jnp.array(kernel_backward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        # self.nethood_forward = jnp.array(kernel_forward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.img_shape = (1,N,N,1)
        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.nethood_forward.shape,  # only ndim matters, not shape 
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit
        self.initialize(self.in_size, self.out_size, self.in_offset, self.out_offset, self.layer_size, self.layer_offset)

    def initialize(self, in_size, out_size, in_offset, out_offset, layer_size, layer_offset):
        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        #init states (integers)
        self.s_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=True)
        states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        passives = jnp.full(layer_size, 2, dtype=self.states_dtype)
        self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        # self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)

        #init weights (random)
        self.w_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=True)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        return hk.get_state("universe")
    
    # def get_cells(self):
    #     return self.universe

    def get_state(self):
        return hk.get_state("universe")
    
    # def clear_cells_vertical(self, universe, offset, after=False):
    #     if after:
    #         result = universe.at[:,offset[1]+1:].set(0)  #clear cells after
    #     else:
    #         result = universe.at[:,:offset[1]].set(0) #clear cells before
    #     return result

    # def set_input_vector(self, inputs):
    #     '''
    #     Set in vector at address with size
    #     '''
    #     return jax.lax.dynamic_update_slice(self.universe, inputs, self.in_offset)
    
    # def set_output_vector(self, outputs):
    #     '''
    #     Write out vector from address with size
    #     '''
    #     return jax.lax.dynamic_update_slice(self.universe, outputs, self.out_offset)
    
    # def get_input_vector(self, input_size):
    #     '''
    #     Read in vector from address with size
    #     '''
    #     return jax.lax.dynamic_slice(self.universe, self.in_offset, input_size)
    
    # def get_output_vector(self, output_size):
    #     '''
    #     Read out vector from address with size
    #     '''
    #     return jax.lax.dynamic_slice(self.universe, self.out_offset, output_size)
    
    def forward(self, weights, biases, states, activate, norm):
        cells = hk.get_state("universe")
        #check backward or forward based on definition of conv in library
        return _forward(cells, weights, biases, states, self.nethood_backward, activate, norm, self.dn)
        # cells = hk.get_state("universe")
        # #get weighted sum of neighbors using convolution
        # cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')

        # if norm:
        #     #norm
        #     cells = jax.nn.standardize(cells, axis=(0,1))

        # #check states
        # #if passive apply activation, else signal state so weight
        # cells = jnp.where(jnp.logical_and(states==1,cells!=0), cells*weights, cells) #weight if signal
        # cells = jnp.where(jnp.logical_and(states==2,cells!=0), activate(cells+biases), cells) #activate if passive

        # return cells

    # def forward_propagate(self, activate=nn.sigmoid, norm=True):
    #     self.universe = self.forward(self.universe, self.weights, self.biases, activate=activate, norm=norm)
    
    def __call__(self, inputs, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        # get params
        states = self.states
        # states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)
        biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #reset cells
        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        # hk.set_state("universe", cells)

        #set input
        cells = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        # hk.set_state("universe", cells)

        #process input
        # cells = jnp.where(jnp.logical_and(states==2,cells!=0), activate(cells+biases), cells) #activate if passive
        hk.set_state("universe", cells)

        #forward
        # for i in range(0, self.depth):
        #     cells = hk.get_state("universe")
        #     #get weighted sum of neighbors using convolution
        #     cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')

        #     if norm:
        #         #norm
        #         cells = jax.nn.standardize(cells, axis=(0,1))

        #     #check states
        #     #if passive apply activation, else signal state so weight
        #     cells = jnp.where(jnp.logical_and(states==1,cells!=0), cells*weights, cells) #weight if signal
        #     if i == self.depth-1:
        #         cells = jnp.where(jnp.logical_and(states==2,cells!=0), final_activate(cells+biases), cells) #final activate if passive
        #     else:
        #         cells = jnp.where(jnp.logical_and(states==2,cells!=0), activate(cells+biases), cells) #activate if passive
        #     hk.set_state("universe", cells)
        for i in range(1, self.depth):
            cells = self.forward(weights, biases, states, activate, norm)
            hk.set_state("universe", cells)
        cells = self.forward(weights, biases, states, final_activate, norm)
        hk.set_state("universe", cells)

        #read output
        return jax.lax.dynamic_slice(cells, self.out_offset, self.out_size)

from functools import partial

@partial(jax.jit, static_argnums=(5,6,7))
def _forward(cells, weights, biases, states, nethood_forward, activate, norm, dn):
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,:,:,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood_forward,  # rhs = OHWI conv kernel tensor
                                        (1,1),  # window strides
                                        'SAME', # padding mode
                                        (1,1),  # lhs/image dilation
                                        (1,1),  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    cells = cells[0,:,:,0] # NHWC

    if norm:
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))

    #check states
    #if passive apply activation, else signal state so weight
    cells = jnp.where(states==1, cells*weights, cells) #weight if signal
    cells = jnp.where(jnp.logical_and(states==2,cells!=0), activate(cells+biases), cells) #activate if passive, more accurate
    # cells = jnp.where(states==2, activate(cells+biases), cells) #activate if passive, faster

    return cells
