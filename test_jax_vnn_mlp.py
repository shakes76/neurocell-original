'''
Von Neumann network that mimics an MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Mapping, Tuple

from universe import Universe

#parameters
N = 8
samples = 120
iterations = 5000
num_classes = 1

#===========
##data
X = np.arange(-30, 30, 0.5, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -10) & (X[:,0] < 10) ] = 1.0

#===========
##create model
# VNN = Universe(N)
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe(N)
    return mlp(batch)

# Make the network and optimiser.
network = hk.without_apply_rng(hk.transform(_forward))
opt = optax.sgd(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), X)
opt_state = opt.init(params)