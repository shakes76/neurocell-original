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
import visualize

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
layers = 16
batch_size = 100
length = 32
channels = 3
depth = layers*2 #1 col of signal cells per layer, channels stacked in last dim so account for it
kernel_size = (7,7,17)
padding = 1*(depth) #+ (kernel_size[2]-1) #depth//2, kernel_size//2 either side of model
N = length + padding #+ depth
print("N:", N, "Epochs:", epochs, "Kernel:", kernel_size)

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
directions = 64 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "directions:", directions, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D(N, sharpen=False)
    mlp.initialize_hyperplane(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Make the network and optimizer. Haiku standard
# decay = 5e-4
# learning_rate = 0.1
# pct_start = 15./epochs #15
# pct_final = 30./epochs #30
# div_factor = 20.
# final_div_factor = 200.
# num_train_steps = (total_batches*epochs)
# lr_schedule = optax.linear_onecycle_schedule(
#         transition_steps=num_train_steps,
#         peak_value=learning_rate,
#         pct_start=pct_start,
#         pct_final=pct_final,
#         div_factor=div_factor,
#         final_div_factor=final_div_factor)
# opt = optax.sgd(lr_schedule, momentum=0.9)
total_steps = (total_batches*epochs)
# clip_factor = 0.01 #AGC parameter
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# AGC should typically be applied BEFORE the optimizer (SGD/Adam) updates are calculated
# or AFTER, depending on interpretation. In NFNets, it clips the raw gradients.
# opt = optax.chain(
#     optax.adamw(learning_rate=schedule),
#     utils.adaptive_gradient_clipping(clipping_threshold=clip_factor),
# )

# Initialize network and optimiser; note we draw an input to get shapes.
dataset_iter = iter(train_loader)
x, y = next(dataset_iter)
x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
# print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1])
opt_state = opt.init(params)
# print(params)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

center_point = (center, center, center)
# plt.imshow(params['universe3_d']['s'][center,...])
# plt.title("Initial States")
# plt.savefig("vnn_3D_hplane_classify_cifar_states_init.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=center_point)

#loss function
@jax.jit
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim
    # print("LOSS SHAPE", labels.shape, logits.shape)

    # return -jnp.mean(labels * jax.nn.log_softmax(logits)) + l2*l2_scaling, state
    # return -jnp.mean(labels * jax.nn.log_softmax(logits)), state

    return optax.softmax_cross_entropy(logits=logits[...,0], labels=labels).mean(), state

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
    updates, opt_state = opt.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(epochs):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for x, y in tqdm(train_loader):
        y = jnp.asarray(y.numpy())

        x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
    total_loss /= total_batches

    if step % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for x, y in tqdm(test_loader):
            # x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())
            # y = jax.nn.one_hot(y, num_classes)

            x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
            predicted = jnp.squeeze(logits, axis=-1)
            predicted = jnp.argmax(predicted[...,0], axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {total_loss}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

jnp.savez("3D_hplane_classify_cifar_cells.npz",
            weights=params['universe3_d']['w'],
            biases=params['universe3_d']['b'],
            filters=params['universe3_d']['f'],
            states=params['universe3_d']['s'],
            N=N, length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

#test performance
correct = 0
total = 0
for x, y in tqdm(test_loader):
    y = jnp.asarray(y.numpy())

    x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits, axis=-1)
    predicted = jnp.argmax(predicted[...,0], axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

plt.imshow(params['universe3_d']['w'][center,...])
plt.title("Final Weights")
plt.savefig("vnn_3D_hplane_classify_cifar_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe3_d']['b'][center,...])
plt.title("Final Biases")
plt.savefig("vnn_3D_hplane_classify_cifar_biases.png", dpi=150)
plt.show()

center_point = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
# plt.imshow(params['universe3_d']['f'][0,center,:,:,0])
# plt.title("Final Filter")
# plt.savefig("vnn_3D_hplane_classify_cifar_filter.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['f'][0,...,0], center=center_point)
visualize.plot_orthogonal_views(params['universe3_d']['f'][1,...,0], center=center_point)

plt.imshow(params['universe3_d']['s'][center,...])
plt.title("Final States")
plt.savefig("vnn_3D_hplane_full_classify_cifar_states.png", dpi=150)
plt.show()

states = params['universe3_d']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states[center,...])
plt.title("Final States (Thresholded)")
plt.savefig("vnn_3D_hplane_full_classify_cifar_states_thres.png", dpi=150)
plt.show()

plt.imshow(state['universe3_d']['universe'][center,...])
plt.title("Final Universe")
plt.savefig("vnn_3D_hplane_full_classify_cifar_universe.png", dpi=150)
plt.show()

print("Layer Norm Parameters\n", params['universe3_d']['scale'], "\n", params['universe3_d']['shift'])

print("END")
