'''
MLP for classifying CIFAR10 dataset
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from tqdm import tqdm, trange #progbar
import time

#dataset
import grains #local

import utils #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
batch_size = 100
length = 32
channels = 3
N = length
print("N:", N, "Epochs:", epochs)

#network IO paths
path = '/home/shakes/'
# path = 'F:/'

#===========
# Data
grains.padding = 0
# Create the dataloader and get total images
train_loader, total_images = grains.create_cifar10_dataloader(
    split="train",
    batch_size=batch_size,
    shuffle=True,
    num_epochs=epochs,
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

test_loader, total_test_images = grains.create_cifar10_dataloader(
    split="test",
    batch_size=batch_size,
    shuffle=False,
    num_epochs=epochs+1,
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

total_batches = total_images // batch_size
total_test_batches = total_test_images // batch_size

# Grain dataloaders yield standard python dictionaries with numpy arrays.
# For JAX, we iterate through the loader and move the arrays to the device.
train_ds = iter(train_loader)
test_ds = iter(test_loader)

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
        net=hk.Linear(512)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(256)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(128)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(32)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(16)(net)
        net=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

def _forward(batch, is_training: bool) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(num_classes)
    return mlp(batch)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

#===========
# Training setup
# Make the network and optimizer. Haiku standard
total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
# batch = next(train_ds)
x = jnp.zeros((batch_size,length,length,channels))
print("INIT SHAPE", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x, is_training=True)
opt_state = opt.init(params)
# print(params)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
@jax.jit
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch, is_training=True)
    # logits = jnp.squeeze(logits, axis=-1) #remove extra dim
    # print("LOSS SHAPE SQUEEZE", labels.shape, logits.shape)

    # return -jnp.mean(labels * jax.nn.log_softmax(logits)) + l2*l2_scaling, state
    # return -jnp.mean(labels * jax.nn.log_softmax(logits)), state

    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        state,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for epoch in range(epochs):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for step in trange(total_batches):
        # Convert numpy arrays to JAX DeviceArrays
        jax_batch = jax.tree.map(jnp.asarray, next(train_ds))

        x = jax_batch['image']
        y = jax_batch['label']

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
        # break
    total_loss /= total_batches

    if epoch % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        # for batch in tqdm(test_loader, total=total_test_batches, desc=f"Testing"):
        for test_step in trange(total_test_batches):
            # Convert numpy arrays to JAX DeviceArrays
            jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

            x = jax_batch['image']
            y = jax_batch['label']

            logits, state_test = network.apply(params, state, x, is_training=False) #if reshaped, use squeeze
            predicted = jnp.argmax(logits, axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Epoch {epoch}, Loss {total_loss}, Test Accuracy {accuracy}]")
    else:
        print(f"[Epoch {epoch}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

#test performance
correct = 0
total = 0
for test_step in trange(total_test_batches):
    # Convert numpy arrays to JAX DeviceArrays
    jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

    x = jax_batch['image']
    y = jax_batch['label']

    logits, state_test = network.apply(params, state, x, is_training=False)
    predicted = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

print("END")
