'''
Von Neumann network that mimics an MLP with a Turing machine structure.

We use cells like a tape
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple

from universe import UniverseLP #local
from data import generate_arithmetic_logic
import forward
import utils #local

utils.jax_status()

#propagate with visuals
@utils.partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_mlp_list_gradients(inputs, offsets, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP, inputs are list of inputs, twos output
    First two outputs are arrays with gradients of each step made, other is the final output
    """
    # gradients_u = jnp.zeros((depth,)+cells.shape, dtype=cells.dtype)
    # gradients_v = jnp.zeros((depth,)+cells.shape, dtype=cells.dtype)

    for input, offset in zip(inputs, offsets): #place inputs to cells
        cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    gradients_u = []
    gradients_v = []
    for i in range(1, depth):
        updated_cells = forward._forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
        u, v = utils.compute_vector_field(cells, updated_cells)
        gradients_u.append(u)
        gradients_v.append(v)
        cells = updated_cells
    updated_cells = forward._forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    u, v = utils.compute_vector_field(cells, updated_cells)
    gradients_u.append(u)
    gradients_v.append(v)
    cells = updated_cells
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    gradients_u = jnp.asarray(gradients_u)
    gradients_v = jnp.asarray(gradients_v)

    return (gradients_u, gradients_v), jax.lax.dynamic_slice(cells, out_offset, out_size)

@utils.partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_mlp_list_visual(inputs, offsets, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP, inputs are list of inputs, twos output
    First output is array with frames of each step made, other is the final output
    """

    for input, offset in zip(inputs, offsets): #place inputs to cells
        cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    frames = []
    for i in range(1, depth):
        cells = forward._forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
        frames.append(cells)
    cells = forward._forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    frames.append(cells)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    return jnp.asarray(frames), jax.lax.dynamic_slice(cells, out_offset, out_size)

##create model
class UniverseVisual(UniverseLP):
    def __init__(self, N, sharpen=True, dtype=jnp.float32, name=None):
        super().__init__(N, sharpen=sharpen, dtype=dtype, name=name)

    def initialize(self, in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=(3,3), directions=1):
        super().initialize(in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=kernel_size, directions=directions)

        self.forward_op = jax.vmap(_propagate_full_mlp_list_visual, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        # print("VISUAL INIT")
        return hk.get_state("universe")

    #call member then defines and stores arrays to visualize as "universe"

#parameters/variables
N = 1 #unused
n = None
bit_depth = None
samples = 64000
batch_size = 128
gap = None
ctrl_gap = None
delta = 0.2
depth = None
# length = None
# num_classes = None
input_size = None
input_offset = None
input_offset_x1 = None
input_offset_x2 = None
ctrl_size = None
ctrl_offset = None
ctrl_bits = None
output_size = None
output_offset = None
# layer_size = None
# layer_offset = None
kernel_size = None
activation = jax.nn.tanh
final_activation = None
directions = None

# filename = "mlp_classify_cells.npz"
filename = "alu_arithmetic_cells.npz"
# filename = "alu_tm_arithmetic_cells.npz"

"""
TM ALU file format

weights=params['universe_lp']['w'],
biases=params['universe_lp']['b'],
filters=params['universe_lp']['f'],
states=params['universe_lp']['s'],
N=N, ctrl_bits=ctrl_bits, bit_depth=bit_depth, n=n, input_size=input_size, output_size=output_size,
input_offset_x1=input_offset_x1, input_offset_x2=input_offset_x2,
crtl_size=ctrl_size, ctrl_offset=ctrl_offset, output_offset=output_offset,
kernel_size=kernel_size,
"""

with np.load(filename, allow_pickle=False) as data:
    weights = data['weights']
    biases = data['biases']
    states = data['states']
    filters = data['filters']
    N=int(data['N'])
    # length=data['length']
    depth=int(data['depth'])
    gap=int(data['gap'])
    ctrl_gap=int(data['ctrl_gap'])
    directions=int(data['directions'])
    # num_classes=data['num_classes']
    input_size=tuple(data['input_size'])
    output_size=tuple(data['output_size'])
    # input_offset=data['input_offset']
    input_offset_x1=tuple(data['input_offset_x1'])
    input_offset_x2=tuple(data['input_offset_x2'])
    output_offset=tuple(data['output_offset'])
    ctrl_size=tuple(data['ctrl_size'])
    ctrl_offset=tuple(data['ctrl_offset'])
    ctrl_bits=int(data['ctrl_bits'])
    bit_depth=int(data['bit_depth'])
    n=int(data['n'])
    # layer_size=data['layer_size']
    # layer_offset=data['layer_offset']
    kernel_size=tuple(data['kernel_size'])

print("N:", N, "Number of Bits:", n, "kernel_size", kernel_size)
print("input_size:", input_size, "output_size:", output_size, "ctrl_size:", ctrl_size)

plt.imshow(weights)
plt.title("Final Weights")
# plt.savefig("vnn_alu_arithmetic_weights.png", dpi=150)
plt.show()

plt.imshow(biases)
plt.title("Final Biases")
# plt.savefig("vnn_alu_arithmetic_biases.png", dpi=150)
plt.show()

plt.imshow(states)
plt.title("Final States")
plt.savefig("vnn_alu_arithmetic_states_final.png", dpi=150)
plt.show()

#===========
#data
X1, X2, C, Y = generate_arithmetic_logic(samples, n, ctrl_bits, batch_size, delta)

# X1 = jnp.asarray(X1)
# X2 = jnp.asarray(X2)
# C = jnp.asarray(C)
# Y = jnp.asarray(Y)

Y_packed = np.packbits(Y,axis=2)
print("Y_packed shape", Y_packed.shape)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseVisual(N, sharpen=False, name="universe_lp")
    mlp.initialize([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Initialize network and optimiser; note we draw an input to get shapes.
x = [X1[0], X2[0], C[0]] #one batch
for index, input in enumerate(x):
    x[index] = x[index][:, :, jnp.newaxis] #fake 2D
print("X0 shape:", x[0].shape)
params, state = network.init(jax.random.PRNGKey(42), x)

params['universe_lp']['w'] = weights
params['universe_lp']['b'] = biases
params['universe_lp']['s'] = states
params['universe_lp']['f'] = filters

#test performance
Y_pred = []
for x1, x2, c, y in zip(X1, X2, C, Y):
    x = [x1, x2, c] #batch of inputs
    for index, input in enumerate(x):
        x[index] = x[index][:, :, jnp.newaxis] #fake 2D

    #forward pass
    logits, state_test = network.apply(params, state, x)

    predictions = jnp.squeeze(logits, axis=-1) #remove extra dim, must match output vector type
    predictions = jnp.round(predictions).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
    break
# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_packed)
print(f"[Final Accuracy {accuracy}]")

#gradients of pass
step = 1
batch_id = 0
y, x = jnp.mgrid[0:N:step, 0:N:step]
for i in range(1, depth):
    print("universe shape:", state_test['universe_lp']['universe'][i-1].shape)
    frame1 = state_test['universe_lp']['universe'][batch_id][i-1]
    frame2 = state_test['universe_lp']['universe'][batch_id][i]
    u, v = utils.compute_vector_field(frame1, frame2)
    # Mask out vectors where the wave amplitude is very low to keep the plot clean
    mask = frame1[::step, ::step] > 0.1
    # Create another 2D array to use for coloring the arrows (e.g., vector magnitude)
    color_array = jnp.hypot(jnp.array(u), jnp.array(v))
    # plot
    plt.quiver(x[mask], y[mask],
                      np.array(u)[::step, ::step][mask],
                      np.array(v)[::step, ::step][mask],
                      color_array[::step, ::step][mask],  # Pass the masked color array here
                      cmap='plasma', scale=25, headwidth=4)
    plt.show()

print("END")
