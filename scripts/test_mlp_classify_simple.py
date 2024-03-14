'''
Simply classify experiment using MLP network
'''
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Mapping, Tuple

import trig #local

#parameters
batch_size = 100
iterations = 4000

#PRNGs
key = jax.random.key(42)

#===========
#paths and names
home_dir = '/home/shakes/'
model_name = "MLP"

#===========
#data
freqs = jnp.array([0.5, 2.0]) #Hz - cycles per second
amplitudes = jnp.array([1.0, 1.0])
num_classes = len(freqs)
time_to_plot = 1 # second
sample_rate = 100 # samples per second
num_samples = sample_rate * time_to_plot
new_key, subkey = jax.random.split(key)
train_data = trig.TrigData(batch_size, freqs, amplitudes, num_samples, sample_rate, new_key)

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
        net=hk.Linear(64)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(32)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(16)(net)
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
# cosine_decay_scheduler = optax.cosine_decay_schedule(1e-4, decay_steps=iterations, alpha=0.95)
# opt = optax.adam(learning_rate=cosine_decay_scheduler)
opt = optax.adam(learning_rate=1e-4)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy including L2 weight regularisation
    '''
    logits = network.apply(params, batch)
    labels = jax.nn.one_hot(labels, num_classes)

    # l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    # return softmax_xent + 1e-4 * l2_loss
    return softmax_xent

# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, batch, labels) -> jnp.ndarray:
    predictions = network.apply(params, batch)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == labels)

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
# params = avg_params = network.init(jax.random.PRNGKey(42), train_ds)
batch, labels = train_data.next()
params = avg_params = network.init(jax.random.PRNGKey(42), batch)
opt_state = opt.init(params)

# Train/eval loop.
for step in range(iterations):
    if step % 100 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        batch, labels = train_data.next()
        train_accuracy = accuracy(avg_params, batch, labels)
        loss_value = loss(params, batch, labels)
        batch, labels = train_data.next()
        test_accuracy = accuracy(avg_params, batch, labels)
        train_accuracy, test_accuracy = jax.device_get(
            (train_accuracy, test_accuracy))
        print(f"[Step {step}, Loss {loss_value}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # Do SGD on a batch of training examples.
    batch, labels = train_data.next()
    params, opt_state = update(params, opt_state, batch, labels)

print("END")
