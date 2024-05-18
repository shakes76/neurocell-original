'''
Universe object for Von Neumann networks
'''
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
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]
kernel_forward =   [[0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
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
        self.key = jax.random.key(42)
        #filters for spatial derivatives
        # self.nethood_backward = jnp.array(kernel_backward, dtype=dtype) # 8 connected kernel
        # self.nethood_forward = jnp.array(kernel_forward, dtype=dtype) # 8 connected kernel
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.initialize() #universe and prev_universe

    def initialize(self):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size, dtype=self.dtype)
        self.key = new_key

        self.save_state()
        
    def initialize_vertical(self, in_size, out_size, in_offset=None, out_offset=None):
        self.universe = jnp.zeros(self.grid_size)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size)
        self.key = new_key

        #setup in/out weights
        self.in_offset = (self.N//4, self.N//8)
        if in_offset:
            self.in_offset = in_offset
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if out_offset:
            self.out_offset = out_offset
        inputs = jnp.ones(in_size, dtype=self.weights.dtype)
        outputs = jnp.ones(out_size, dtype=self.weights.dtype)
        weights = self.clear_cells_vertical(self.weights, self.in_offset, after=False)
        self.weights = jax.lax.dynamic_update_slice(weights, inputs, self.in_offset)
        weights = self.clear_cells_vertical(self.weights, self.out_offset, after=True)
        self.weights = jax.lax.dynamic_update_slice(weights, outputs, self.out_offset)

        self.save_state()

    def get_cells(self):
        return self.universe
        
    def get_weights(self):
        return self.weights
    
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
    
    def forward(self, cells, activate=nn.sigmoid, norm=True):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')
        #apply weights
        cells = cells * self.weights
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)
    
    def forward_propagate(self, activate=nn.sigmoid, norm=True):
        self.universe = self.forward(self.universe, activate=activate, norm=norm)

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
        
    def loss_perturb(self, cells, deltas, labels, inputs, in_offset, output_size, out_offset, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        '''
        Cross entropy 
        We use the perturbation trick, so get the layer by layer grad values for the overall loss
        See https://github.com/google/jax/discussions/5336#discussioncomment-269983 
        '''
        depth = len(deltas)
        #set input
        # cells = self.clear_cells_vertical(cells, in_offset, after=False)
        cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)
        #forward
        activations = []
        for i in range(1, depth):
            cells = self.forward(cells, activate=activate, norm=norm)
            cells = cells + deltas[i-1] #have grad return layer grad
            activations.append(cells) #save layer grad
        cells = self.forward(cells, activate=final_activate, norm=norm)
        cells = cells + deltas[depth-1]
        activations.append(cells)
        # print("activations size:", len(activations))

        #read output
        logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
        loss_value = -1 * jnp.mean(labels * jnp.log(logits + self.epsilon) + (1. - labels) * jnp.log(1. - logits + self.epsilon)) #CE

        return loss_value, activations

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
        return jax.value_and_grad(self.loss_perturb, (0,1), has_aux=True)(self.universe, deltas, labels, inputs, in_offset, out_size, out_offset, norm=False)

    def value_and_grad_update(self, lr, depth, labels, inputs, in_offset, out_size, out_offset):
        (self.loss_value, self.layer_grads), self.grads_output = self.value_and_grad_perturb(depth, labels, inputs, in_offset, out_size, out_offset)
        # print("loss_value", self.loss_value)
        # print("layer grads", len(self.layer_grads), self.layer_grads[0].shape)

        #compose per layer grads
        # self.grads = self.layer_grads[1]
        self.grads = jnp.zeros_like(self.universe)
        for i in range(1,depth):
            self.grads = self.grads.at[:,self.out_offset[1]-i].set(self.layer_grads[depth-i-1][:,self.out_offset[1]-i])
        # self.grads = self.grads.at[:,self.in_offset[1]].set(self.grads_output[0][:,self.in_offset[1]])

        #apply corrections
        self.weights = self.weights - lr*self.grads
        return self.loss_value, self.grads, self.weights

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
        '''self.in_offset = (self.N//4, self.N//8)
        if in_offset:
            self.in_offset = in_offset
        pred_inputs = jax.lax.dynamic_slice(self.universe, self.in_offset, input_size)
        scaling = lr*abs(inputs/pred_inputs) #only mag
        # if scaling > 1.0:
        #     scaling = jnp.minimum(1.5, scaling)
        # unity = jnp.ones((1,1), dtype=self.universe.dtype)
        zero = jnp.zeros((1,1), dtype=self.universe.dtype)
        self.universe = jax.lax.dynamic_update_slice(self.universe, zero, self.in_offset)
        # if scaling > 1.0:
        #     self.universe = scaling*self.universe
        self.universe = scaling*self.universe
        print("Scaling:", scaling)'''
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
        # cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)

    def backward_propagate(self, cells, activate=nn.sigmoid):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_backward, mode='same')
        #norm
        # cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)
    
    def __call__(self, x, in_offset, output_size, out_offset, activate=nn.sigmoid):
        #input
        net = self.set_input_vector(x, in_offset)
        #forward pass
        self.universe = self.forward_propagate(net, activate)
        #read output
        return self.get_output_vector(output_size, out_offset)
