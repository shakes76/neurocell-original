'''
Von Neumann network that mimics an MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Tuple

from universe import UniverseMLP #local
import utils #local

utils.jax_status()

#parameters
N = 7
epochs = 1
samples = 120
length = 1
num_classes = 1
print("N:", N, "Epochs:", epochs)

#===========
##data
X = np.arange(-1.5, 1.5, 0.025, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -0.5) & (X[:,0] < 0.5) ] = 1.0

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
layer_size = (5,1)
layer_offset = (1,3)
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
    mlp = UniverseMLP(N, input_size, output_size, input_offset, output_offset, layer_size, layer_offset)
    return mlp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform(_forward))
opt = optax.sgd(learning_rate=0.1)
# opt = optax.adam(learning_rate=0.001)

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), X[0].reshape(-1,1))
opt_state = opt.init(params)
print(params)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    logits = network.apply(params, batch)
    return -1 * jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1. - logits))

# @jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value

# Train/eval loop.
for step in range(epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        params, opt_state, value = update(params, opt_state, x.reshape(-1,1), y.reshape(-1,1))
        loss_value += value

    if step % 100 == 0:
        print(f"[Step {step}, Loss {loss_value}]")
        # Periodically evaluate on train & test sets.
        Y_pred = []
        for x, y in zip(X, Y):
            Y_pred.append(network.apply(params, x.reshape(-1,1))[0])
        plt.plot(X, Y_pred)
        plt.plot(X, Y, 'm')
        plt.xlabel('X')
        plt.show()

plt.imshow(params['universe_mlp']['w'])
plt.title("Final Network")
plt.show()

print("END")