'''
Von Neumann network that mimics an MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Mapping, Tuple

from universe import Universe

#parameters
N = 8
samples = 120
iterations = 400
num_classes = 1
print("N:", N, "iterations:", iterations)

#===========
##data
X = np.arange(-30, 30, 0.5, dtype=np.float32).reshape(-1,1)
Y = np.zeros(samples, dtype=np.float32)
# print(X[0].reshape(-1,1))

#square wave
Y[ (X[:,0] >= -10) & (X[:,0] < 10) ] = 1.0

#locations
#MLP with 1 input and 1 output, one layer inbetween
# input_size = (1,1)
input_offset = (3,2)
output_size = (num_classes,1)
output_offset = (3,4)

#===========
##create model
def _forward(batch, in_offset, out_size, out_offset, act=jax.nn.sigmoid) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe(N)
    return mlp(batch, in_offset, out_size, out_offset, act)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform(_forward))
# opt = optax.sgd(learning_rate=0.01)
opt = optax.adam(learning_rate=0.001)

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), X[0].reshape(-1,1), input_offset, output_size, output_offset)
opt_state = opt.init(params)

#loss function
def loss(params: hk.Params, batch, labels, in_offset, out_size, out_offset) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    logits = network.apply(params, batch, in_offset, out_size, out_offset)
    return -1 * jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1. - logits))

# @jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch,
        labels,
        in_offset, 
        out_size, 
        out_offset
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch, labels, in_offset, out_size, out_offset)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Train/eval loop.
for step in range(iterations+1):
    # Do SGD on a batch of training examples.
    for x, y in zip(X, Y):
        params, opt_state = update(params, opt_state, x.reshape(-1,1), y.reshape(-1,1), input_offset, output_size, output_offset)

    if step % 400 == 0:
        loss_value = 0
        for x, y in zip(X, Y):
            loss_value += loss(params, x.reshape(-1,1), y.reshape(-1,1), input_offset, output_size, output_offset)
        print(f"[Step {step}, Loss {loss_value}]")
        # Periodically evaluate on train & test sets.
        Y_pred = []
        for x, y in zip(X, Y):
            Y_pred.append(network.apply(params, x.reshape(-1,1), input_offset, output_size, output_offset)[0])
        plt.plot(X, Y_pred)
        plt.plot(X, Y, 'm')
        plt.xlabel('X')
        plt.show()

plt.imshow(params['universe']['w'])
plt.title("Final Network")
plt.show()

print("END")