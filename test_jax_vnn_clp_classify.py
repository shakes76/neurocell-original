'''
Von Neumann network that mimics an MLP
'''
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from sklearn import datasets

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
epochs = 20
architecture_filename = "patterns/Sprite-CLP13.png"
start_state = 1 #set None for random state

#===========
##data
data_dict = datasets.load_wine(return_X_y=False)
X = data_dict['data']
Y = data_dict['target']
length = len(data_dict['feature_names'])
samples = len(data_dict['target'])
num_classes = len(data_dict['target_names'])
print("Samples:", samples, "Input Length:", length, "Classes:", num_classes)

#load network architecture
N, input_coords, neuron_coords, output_coords = utils.load_sprite(architecture_filename)
input_coords = jnp.array(input_coords)
neuron_coords = jnp.array(neuron_coords)
output_coords = jnp.array(output_coords)
print("N:", N, "Epochs:", epochs)

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
kernel_size = (3,3)
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
#depth
in_row_coord = input_coords[:, 0].min() #top most in
out_row_coord = output_coords[:, 0].min() #top most out
in_col_coord = input_coords[:, 1].min() #far left in
out_col_coord = output_coords[:, 1].max() #far right out
depth_row = out_row_coord - in_row_coord
depth_col = out_col_coord - in_col_coord
depth = max(depth_row, depth_col)
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    clp = UniverseLP(N)
    clp.initialize_clp(input_coords, output_coords, neuron_coords, depth, kernel_size)
    return clp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.001)

# Initialize network and optimiser; note we draw an input to get shapes.
params, state = network.init(jax.random.PRNGKey(42), X[0])
opt_state = opt.init(params)
# print(params)
# print(state)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)

    #CE loss
    logits, state = network.apply(params, state, batch)
    ce = -jnp.sum(labels * jax.nn.log_softmax(logits))

    #L2 norm of weights loss
    filters = params['universe_lp']['f']
    l2 = jnp.mean(filters**2)

    return ce + l2*l2_scaling, state

# @jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        state,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value
        # break

    if step % 10 == 0:
        Y_pred = []
        for x, y in zip(X, Y):
            logits, state_test = network.apply(params, state, x)
            Y_pred.append(jnp.squeeze(logits, axis=-1))
        Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        accuracy = jnp.mean(Y_pred == Y)
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("clp_classify_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=state['universe_lp']['states'], 
            N=N, length=length, num_classes=num_classes, input_coords=input_coords, output_coords=output_coords,
            neuron_coords=neuron_coords, kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_clp_classify_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_clp_classify_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_clp_classify_filter.png", dpi=75)
plt.show()

plt.imshow(state['universe_lp']['states'])
plt.title("Final States")
plt.savefig("vnn_clp_classify_states.png", dpi=75)
plt.show()

#function plot
Y_pred = []
for x, y in zip(X, Y):
    logits, state_test = network.apply(params, state, x)
    Y_pred.append(logits)
Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
accuracy = jnp.mean(Y_pred == Y)
print(f"[Final Accuracy {accuracy}]")

print("END")