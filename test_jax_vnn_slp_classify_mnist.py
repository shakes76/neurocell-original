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

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
epochs = 4
num_classes = 10
batch_size = 5
length = 28*28
padding = 8
N = length//2+padding
print("N:", N, "Epochs:", epochs)

#network IO paths
path = 'F:/'

#===========
#transform preproc
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

##data
trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #num_workers=6

testset = torchvision.datasets.MNIST(root=path+'data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) #num_workers=6

num_classes = 10

#locations
#SLP with 784 inputs and 10 outputs, learns num of layers inbetween
#setup network position
center = N//2
offset = (padding//2,center-length//8) #center-length//2=padding, center-top
# input_top = center-length//4
# output_top = center-num_classes//2
# output_left = input_left+total_layers-1 #dont include out obv
# print("input_top", input_top, "input_left", input_left, "output_top", output_top, "output_left", output_left)
#setup parameters and lengths
input_size = (length,1)
input_offset = offset
output_size = (num_classes,1)
output_offset = (offset[0]+length//4-num_classes//2, offset[1]+length//4+2) #shift by half of mid and top
kernel_size = (5,5)
activation = jax.nn.tanh
final_activation = None
#determine dim with largest depth to propagate forward
depth = max(output_offset[1]-input_offset[1], output_offset[0]-input_offset[0])
total_batches = len(train_loader)
print("depth:", depth, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N)
    mlp.initialize_slp_full(input_size, output_size, input_offset, output_offset, kernel_size)
    return mlp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.001)

# Initialize network and optimiser; note we draw an input to get shapes.
dataset_iter = iter(train_loader)
x, y = next(dataset_iter)
x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
x = jnp.reshape(x, (x.shape[0], length, 1))
# print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1])
opt_state = opt.init(params)
# print(params)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim
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

        x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
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

            x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
            x = jnp.reshape(x, (x.shape[0], length, 1)) #image to vector

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
            predicted = jnp.squeeze(logits, axis=-1)
            predicted = jnp.argmax(jnp.array(predicted), axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {total_loss}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("mlp_classify_mnist_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=params['universe_lp']['s'], 
            N=N, length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

#test performance
correct = 0
total = 0
for x, y in tqdm(test_loader):
    x = jnp.asarray(x.numpy())
    y = jnp.asarray(y.numpy())

    x = jnp.transpose(x, (0, 2, 3, 1)) #channel last
    x = jnp.reshape(x, (x.shape[0], length, 1)) #image to vector

    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits, axis=-1)
    predicted = jnp.argmax(jnp.array(predicted), axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_mlp_classify_mnist_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_mlp_classify_mnist_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_mlp_classify_mnist_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_mlp_full_classify_mnist_states.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_mlp_full_classify_mnist_states_thres.png", dpi=75)
plt.show()

print("END")