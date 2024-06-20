'''
Simple MLP trained as a classify network for a simple dataset - MNIST
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Tuple

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#parameters
iterations = 2

#===========
##data
# Load data from https://www.openml.org/d/554
X_raw, Y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='pandas')
Y_raw = Y_raw.astype(np.int32)
length = X_raw.shape[1]
samples = X_raw.shape[0]
num_classes = 10
print("Data Shape:", X_raw.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes, "Classes:", num_classes)
print("Y raw:", Y_raw.dtype, Y_raw[:10])

#scaling
scaler = StandardScaler()
X_preproc = scaler.fit_transform(X_raw)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

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
        net=hk.Linear(256)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(128)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(32)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(num_classes)
    return mlp(batch)

# Make the network and optimiser.
network = hk.without_apply_rng(hk.transform(_forward))
opt = optax.sgd(learning_rate=0.01)
# opt = optax.adam(learning_rate=0.001)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits = network.apply(params, batch)
    return -jnp.sum(labels * jax.nn.log_softmax(logits))

@jax.jit
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

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), X)
opt_state = opt.init(params)

# Train/eval loop.
for step in range(iterations+1):
    # Do SGD on a batch of training examples.
    for x, y in zip(X, Y):
        params, opt_state, loss_value = update(params, opt_state, x, y)

    if step % 1 == 0:
        # Periodically evaluate on train & test sets.
        Y_pred = network.apply(params, X)
        accuracy = jnp.mean(jnp.argmax(Y_pred, axis=-1) == Y)
        print(f"[Step {step}, Loss {loss_value}, Train Accuracy {accuracy}]")

#test performance
Y_pred = []
for x, y in zip(X_test, Y_test):
    logits = network.apply(params, x)
    Y_pred.append(logits)
Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
accuracy = jnp.mean(Y_pred == Y_test)
print(f"[Final Accuracy {accuracy}]")

print("END")
