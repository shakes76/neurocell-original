'''
Simple MLP trained as a coordinate transform to approximate a mathematical function
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Tuple

#parameters
samples = 120
iterations = 2000
num_classes = 1

#===========
##data
X = np.arange(0, 60, 0.5).reshape(-1, 1)
Y = np.zeros(samples)

#square wave
Y[ (X[:,0] >= 20) & (X[:,0] < 40) ] = 1.0

plt.plot(X, Y)
plt.show()

#===========
##create model
class MLP(hk.Module):
    '''
    Multi-layer perceptron
    Applies flattening to input and returns logits
    '''
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        '''
        MLP Network
        '''
        #define layers
        net=hk.Flatten()(x)
        net=hk.Linear(2)(net)
        net=jax.nn.sigmoid(net)
        net=hk.Linear(self.output_size)(net)
        net=jax.nn.sigmoid(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(num_classes)
    return mlp(batch)

# Make the network and optimiser.
network = hk.without_apply_rng(hk.transform(_forward))
# opt = optax.sgd(learning_rate=0.01)
opt = optax.adam(learning_rate=1e-4)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    logits = network.apply(params, batch)
    return -1 * jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1. - logits))
    # return -jnp.sum(labels * jnp.log(logits))

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Initialize network and optimiser; note we draw an input to get shapes.
params = avg_params = network.init(jax.random.PRNGKey(42), X[0])
opt_state = opt.init(params)

# Train/eval loop.
for step in range(iterations):
    # Do SGD on a batch of training examples.
    for x, y in zip(X, Y):
        params, opt_state = update(params, opt_state, X, Y)

    if step % 500 == 0:
        loss_value = loss(params, X, Y)
        print(f"[Step {step}, Loss {loss_value}]")
        # Periodically evaluate on train & test sets.
        Y_pred = network.apply(params, X)
        plt.plot(X, Y_pred)
        plt.plot(X, Y, 'm')
        plt.xlabel('X')
        plt.show()

print("END")
