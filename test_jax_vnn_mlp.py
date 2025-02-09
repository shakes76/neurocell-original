'''
Von Neumann network that mimics an MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
N = 7
epochs = 200
samples = 120
batch_size = 12
num_batches = samples//batch_size
length = 1
num_classes = 1
epsilon = 1e-8
print("N:", N, "Epochs:", epochs)

#===========
##data
# reshapes to get 2D arrays 
X = np.arange(-1.5, 1.5, 0.025, dtype=np.float32).reshape(-1,1)
Y = np.zeros(samples, dtype=np.float32).reshape(-1,1)
print("X shape", X.shape, "Y shape", Y.shape)

#square wave, ignoring extra dim
Y[ (X[:,0] >= -0.5) & (X[:,0] < 0.5), 0 ] = 1.0

#reshape to number of batches
X = np.reshape(X, (num_batches,batch_size,1))
Y = np.reshape(Y, (num_batches,batch_size,1))

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
depth = output_offset[1]-input_offset[1]
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N)
    mlp.initialize_mlp(input_size, output_size, input_offset, output_offset)
    return mlp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
# print("X0 shape:", x.shape)
x = x[:, jnp.newaxis] #fake 2D
print("X0 shape:", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-4) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    # labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim
    ce = -1 * jnp.mean(labels * jnp.log(logits + epsilon) + (1 - labels) * jnp.log(1. - logits + epsilon))

    #L2 norm of weights loss
    # filters = params['universe_lp']['f']
    # l2 = jnp.mean(filters**2)

    # return ce + l2*l2_scaling, state
    return ce, state
    # return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state
    # return optax.l2_loss(predictions=logits, targets=labels).mean(), state

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
        x = x[:, jnp.newaxis] #fake 2D
        # print(x.shape)
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value
        # break

    print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}]")
    # if step % 100 == 0:
    #     # Periodically evaluate on train & test sets.
    #     Y_pred = []
    #     for x, y in zip(X, Y):
    #         logits, state_test = network.apply(params, state, x.reshape(-1,1))
    #         Y_pred.append(logits[0])
    #     plt.plot(X, Y_pred)
    #     plt.plot(X, Y, 'm')
    #     plt.xlabel('X')
    #     plt.show()
    # break
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("mlp_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=params['universe_lp']['s'], 
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_mlp_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_mlp_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_mlp_states.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_mlp_filter.png", dpi=75)
plt.show()

#function plot
fig = plt.figure()
for x, y in zip(X, Y):
    x = jnp.squeeze(x, axis=-1) #remove extra dim
    y = jnp.squeeze(y, axis=-1) #remove extra dim
    plt.plot(x, y, 'm')
for x, y in zip(X, Y):
    x = x[:, jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.squeeze(logits, axis=-1) #remove extra dim
    # for pred in predictions:
    # for val, pred in zip(x, predictions): # samples in batch
    x = jnp.squeeze(x, axis=-1) #remove extra dim
    plt.plot(x, predictions, 'b')
plt.xlabel('X')
plt.savefig("vnn_mlp_plot.png", dpi=150)
plt.show()

print("END")