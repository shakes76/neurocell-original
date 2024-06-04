'''
Unit test for effect of filters
'''
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.signal as jsignal
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from functools import partial

import filters #local
import utils #local

utils.jax_status()

#parameters
N = 7
epochs = 10
samples = 120
length = 1
num_classes = 1
epsilon = 1e-6
lr = 0.1

nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel

#===========
##data
X = np.arange(-1.5, 1.5, 0.025, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -0.5) & (X[:,0] < 0.5) ] = 1.0

#funcs
@partial(jax.jit, static_argnums=(5,6))
def forward(cells, weights, biases, states, nethood, activate=jax.nn.sigmoid, norm=True):
    #get weighted sum of neighbors using convolution
    cells = jsignal.convolve2d(cells, nethood, mode='same')

    if norm:
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))
    #check states
    #if passive apply activation, else signal state so weight
    cells = jnp.where(states==1, cells*weights, cells) #weight if signal
    cells = jnp.where(jnp.logical_and(states==2,cells!=0), activate(cells+biases), cells) #activate if passive

    return cells

def forward_propagate(universe, weights, biases, states, activate=jax.nn.sigmoid, norm=True):
    return forward(universe, weights, biases, states, nethood_forward, activate=activate, norm=norm)

def loss_perturb(params, cells, states, depth, labels, inputs, in_offset, output_size, out_offset, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, norm=True):
    '''
    Cross entropy 
    We use the perturbation trick, so get the layer by layer grad values for the overall loss
    See https://github.com/google/jax/discussions/5336#discussioncomment-269983 
    '''
    weights, biases = params
    #set input
    # cells = self.clear_cells_vertical(cells, in_offset, after=False)
    cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)
    #forward
    for i in range(1, depth):
        cells = forward(cells, weights, biases, states, nethood_forward, activate=activate, norm=norm)
        # cells = cells + deltas[i-1] #have grad return layer grad
    cells = forward(cells, weights, biases, states, nethood_forward, activate=final_activate, norm=norm)
    # cells = cells + deltas[depth-1]
    # print("activations size:", len(activations))

    #read output
    logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
    loss_value = -1 * jnp.mean(labels * jnp.log(logits) + (1. - labels) * jnp.log(1. - logits)) #CE

    return loss_value

def value_and_grad_update(params, cells, states, lr, depth, labels, inputs, in_offset, out_size, out_offset, activate=jax.nn.tanh, final_activate=jax.nn.sigmoid, norm=True):
    loss_value, grads_output = jax.value_and_grad(loss_perturb)(params, cells, states, depth, labels, inputs, in_offset, out_size, out_offset, activate, final_activate, norm)
    # print("loss_value", self.loss_value)
    # print("layer grads", len(self.layer_grads), self.layer_grads[0].shape)
    # print("grads", len(self.grads_output), self.grads_output)
    # print("grads", len(self.grads_output[0]), self.grads_output[0])
    # print("grads", len(self.grads_output[1]), self.grads_output[1])

    #compose per layer grads
    # grads = grads_output[0]
    # bias_grads = grads_output[1]

    #apply corrections
    # weights, biases = params
    # weights = weights - lr*grads
    # biases = biases - lr*bias_grads
    # new_params = [weights, biases]
    new_params = jax.tree_map(
      lambda param, g: param - g * lr, params, grads_output)

    return loss_value, new_params

def clear_cells_vertical(universe, offset, after=False):
    '''
    Clears cells upto and including offset positions
    '''
    if after:
        result = universe.at[:,offset[1]:].set(0)  #clear cells after
    else:
        result = universe.at[:,:offset[1]+1].set(0) #clear cells before
    return result

def initialize_vertical(N, in_size, out_size, in_offset, out_offset, dtype=jnp.float32, states_dtype=jnp.uint8):
    grid_size = (N,N)
    universe = jnp.zeros(grid_size, dtype=dtype)
    states = jnp.ones(grid_size, dtype=states_dtype)

    key = jax.random.key(42)
    new_key, subkey = jax.random.split(key)
    weights = 1.0 * jax.random.normal(subkey, grid_size)
    biases = jnp.zeros(grid_size, dtype=dtype)

    #setup in/out weights
    in_passives = jnp.full(in_size, 2, dtype=states_dtype)
    out_passives = jnp.full(out_size, 2, dtype=states_dtype)
    inputs = jnp.ones(in_size, dtype=weights.dtype)
    outputs = jnp.ones(out_size, dtype=weights.dtype)
    weights = clear_cells_vertical(weights, in_offset, after=False)
    weights = jax.lax.dynamic_update_slice(weights, inputs, in_offset)
    states = clear_cells_vertical(states, in_offset, after=False)
    states = jax.lax.dynamic_update_slice(states, in_passives, in_offset)
    weights = clear_cells_vertical(weights, out_offset, after=True)
    weights = jax.lax.dynamic_update_slice(weights, outputs, out_offset)
    states = clear_cells_vertical(states, out_offset, after=True)
    states = jax.lax.dynamic_update_slice(states, out_passives, out_offset)

    return universe, weights, biases, states

def set_layer_vertical(states, layer_size, offset=None, states_dtype=jnp.uint8):
    passives = jnp.full(layer_size, 2, dtype=states_dtype)
    states = jax.lax.dynamic_update_slice(states, passives, offset)
    return states

def set_input_vector(universe, inputs, offset):
    '''
    Set in vector at address with size
    '''
    cells = clear_cells_vertical(universe, offset, after=False)
    universe = jax.lax.dynamic_update_slice(cells, inputs, offset)
    return universe

def reset_cells(N, dtype=jnp.float32):
    grid_size = (N,N)
    return jnp.zeros(grid_size, dtype=dtype)

#===========
#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
layer_size = (5,1)
layer_offset = (1,3)
depth = output_offset[1]-input_offset[1]
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
print("depth:", depth)

# Initialize model
universe, weights, biases, states = initialize_vertical(N, input_size, output_size, input_offset, output_offset)
states = set_layer_vertical(states, layer_size, layer_offset)
# print(weights)
# print(biases)
# print(states)

#parameters to optimize
params = [weights, biases]
# final_params = {"w":weights, "b":biases}

start = time.time() #time generation
for epoch in range(1, epochs+1):
    Y_pred = []
    for iteration, (x, y) in enumerate(zip(X, Y)):
        #set output
        universe = set_input_vector(universe, x.reshape(-1,1), input_offset)
        
        #grad
        loss_value, params = value_and_grad_update(params, universe, states, lr, depth, y.reshape(-1,1), x.reshape(-1,1), input_offset, output_size, output_offset, activation, final_activation, False)
        # forward = model.get_cells() #for plot
        # universe = reset_cells(N) #reset to zero

    # for iteration, (x, y) in enumerate(zip(X, Y)): #without this we are leaking data, fits for each point used.
        #forward pass to check output
        weights, biases = params
        #set output
        universe = set_input_vector(universe, x.reshape(-1,1), input_offset)
        for i in range(1, depth):
            universe = forward_propagate(universe, weights, biases, states, activate=activation, norm=False)
        universe = forward_propagate(universe, weights, biases, states, activate=final_activation, norm=False)
        output2 = jax.lax.dynamic_slice(universe, output_offset, output_size)
        Y_pred.append(output2.reshape(1))
        universe = reset_cells(N) #reset to zero
        
        # if iteration%10 == 0:
        #     print("i:", iteration, "X", x, "Y:", y, "loss:", loss_value)
        # break
    print(">> Epoch:", epoch, "Loss:", loss_value)
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

#save models
# model.save("cells.npz")

#set output
weights, biases = params
Y_pred_test = []
for iteration, (x, y) in enumerate(zip(X, Y)):
    universe = set_input_vector(universe, x.reshape(-1,1), input_offset)
    for i in range(1, depth):
        universe = forward_propagate(universe, weights, biases, states, activate=activation, norm=False)
    universe = forward_propagate(universe, weights, biases, states, activate=final_activation, norm=False)
    output2 = jax.lax.dynamic_slice(universe, output_offset, output_size)
    Y_pred_test.append(output2.reshape(1))
    universe = reset_cells(N) #reset to zero

plt.imshow(universe)
plt.title("Final Result")
plt.savefig("Final_Result.png", dpi=150)
#plt.show()

plt.imshow(weights)
plt.title("Final Weights")
plt.savefig("Final_Weights.png", dpi=150)
plt.show()

plt.imshow(biases)
plt.title("Final Biases")
plt.savefig("Final_Biases.png", dpi=150)
plt.show()

plt.imshow(states)
plt.title("Final States")
plt.savefig("Final_States.png", dpi=150)
plt.show()

#function plot
fig = plt.figure()
plt.plot(X, Y_pred_test, "r")
plt.plot(X, Y_pred)
plt.plot(X, Y, 'm')
plt.xlabel('X')
plt.savefig("Plot.png", dpi=150)
plt.show()
