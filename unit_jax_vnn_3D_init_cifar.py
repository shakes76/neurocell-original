'''
Von Neumann network that creates a 3D MLP for classifying CIFAR10 dataset

Uses hyperplanes (n-1 D) convs per layer as the forward operator, more efficient.
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

from universe import Universe3D #local
import utils #local
import visualize #local

utils.jax_status()

#parameters
epochs = 1
num_classes = 10
layers = 4
batch_size = 100
length = 32
channels = 3
depth = layers*2+1 #1 col of signal cells per layer
kernel_size = (9,9,7)
padding = 1*(depth-1) + (kernel_size[0]-1) #depth//2, kernel_size//2 either side of model
N = length + padding #+ depth
print("N:", N, "Epochs:", epochs)

#network IO paths
path = '/home/shakes/'
# path = 'F:/'

#===========
#transform preproc
#lambda for channel last
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=path+'data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #num_workers=6

testset = torchvision.datasets.CIFAR10(
    root=path+'data/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False) #num_workers=6

num_classes = 10
total_batches = len(train_loader)

#locations
#MLP with 784 inputs and 10 outputs, 3 layers inbetween
#setup network position
center = N//2
input_top = center-length//2
input_left = center-length//2
input_inner = center-(depth-1)//2
output_top = center-num_classes//2
output_left = center
output_inner = input_inner+depth+1 #include out layer
# print("input_top", input_top, "input_left", input_left, "output_top", output_top, "output_left", output_left)
#setup parameters and lengths
input_size = (length,length,channels)
input_offset = (input_top,input_left,input_inner)
output_size = (num_classes,1,1)
output_offset = (output_top,output_left,output_inner)
activation = jax.nn.leaky_relu
final_activation = jax.nn.leaky_relu
directions = 1024 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "directions:", directions, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D(N, sharpen=True)
    #use depth=0 so no propagation for init vis
    mlp.initialize_hyperplane(input_size, output_size, input_offset, output_offset, depth=depth, kernel_size=kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Initialize network and optimiser; note we draw an input to get shapes.
dataset_iter = iter(train_loader)
x, y = next(dataset_iter)
x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1])
# print(params)

center_point = (center, center, center)
# plt.imshow(params['universe3_d']['s'][center,...])
# plt.title("Initial States")
# # plt.savefig("vnn_3D_hplane_classify_cifar_states_init.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=center_point)

center_point = (input_top,input_left,input_inner)
# plt.imshow(state['universe3_d']['universe'][...,input_inner])
# plt.title("Initial Universe")
# # plt.savefig("vnn_3D_hplane_full_classify_cifar_universe.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=center_point)

# Train/eval loop.
start = time.time() #time generation
for step in range(epochs):
    # total_loss = 0
    # Do SGD on a batch of training examples.
    for x, y in tqdm(train_loader):
        y = jnp.asarray(y.numpy())

        x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

        logits, state = network.apply(params, state, x) #if reshaped, use squeeze

        center_point = (input_top,input_left,input_inner)
        # plt.imshow(state['universe3_d']['universe'][...,input_inner])
        # plt.title("Initial Universe")
        # # plt.savefig("vnn_3D_hplane_full_classify_cifar_universe.png", dpi=150)
        # plt.show()
        visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=center_point)

        # params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        # total_loss += loss_value/batch_size
        break
    # total_loss /= total_batches

    if step % 1 == 0:
        # Periodically evaluate on train & test sets.
        # correct = 0
        # total = 0
        for x, y in tqdm(test_loader):
            # x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())
            # y = jax.nn.one_hot(y, num_classes)

            x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze

            center_point = (input_top,input_left,input_inner)
            # plt.imshow(state['universe3_d']['universe'][...,input_inner])
            # plt.title("Initial Universe")
            # # plt.savefig("vnn_3D_hplane_full_classify_cifar_universe.png", dpi=150)
            # plt.show()
            visualize.plot_orthogonal_views(state_test['universe3_d']['universe'], center=center_point)

            # predicted = jnp.squeeze(logits, axis=-1)
            # predicted = jnp.argmax(predicted[...,0], axis=-1)
            # correct += jnp.sum(predicted == y)
            # total += y.shape[0]
            break
        # accuracy = 100 * correct / total
        # print(f"[Step {step}, Loss {total_loss}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

print("END")
