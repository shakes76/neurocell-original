'''
Von Neumann network that mimics an MLP for classifying MNIST dataset
'''
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm #progbar
import time

#dataset
import torch
import torchvision
import torchvision.transforms as transforms

import utils #local

utils.jax_status()

#parameters
N = 1024
epochs = 8
num_classes = 10
layers = 4
batch_size = 64
length = 28*28
print("N:", N, "Epochs:", epochs)

#network IO paths
path = '/home/shakes/'

#===========
#transform preproc
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Lambda(lambda x: x.permute(1, 2, 0))])

##data
trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #num_workers=6

testset = torchvision.datasets.MNIST(root=path+'data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) #num_workers=6

num_classes = 10
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
total_batches = len(train_loader)
print("total batches:", total_batches)

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

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
dataset_iter = iter(train_loader)
x, y = next(dataset_iter)
# x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
x = jnp.reshape(x.numpy(), (x.shape[0], length, 1))
# print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1])
opt_state = opt.init(params)
# print(params)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch)
    # print("LOSS SHAPE", labels.shape, logits.shape)
    # print("LOSS", jnp.mean(labels * jax.nn.log_softmax(logits)).shape, logits.shape)

    #L2 norm of weights loss
    # filters = params['universe_lp']['f']
    # l2 = jnp.mean(filters**2)

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
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(epochs+1):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for x, y in tqdm(train_loader):
        x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())

        # x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
        # print("TRAIN SHAPE", x.shape)
        x = jnp.reshape(x, (x.shape[0], length, 1)) #image to vector
        # print("TRAIN SHAPE", x.shape)

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
    total_loss /= total_batches

    if step % 2 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for x, y in tqdm(test_loader):
            x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())

            # x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
            x = jnp.reshape(x, (x.shape[0], length, 1)) #image to vector

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
            predicted = jnp.argmax(logits, axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {total_loss}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

#test performance
correct = 0
total = 0
for x, y in tqdm(test_loader):
    x = jnp.asarray(x.numpy())
    y = jnp.asarray(y.numpy())

    # x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
    x = jnp.reshape(x, (x.shape[0], length, 1)) #image to vector

    logits, state_test = network.apply(params, state, x)
    predicted = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

print("END")
