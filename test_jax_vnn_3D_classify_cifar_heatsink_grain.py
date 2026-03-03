'''
Von Neumann network that creates a 3D MLP for classifying CIFAR10 dataset

Uses full 3D convs as the forward operator, expensive.

You may need to disable the XLA conv picker using env variables:
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
'''
# import os

# # Set the XLA flags here, before any JAX import
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# os.environ['XLA_FLAGS'] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

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
import tensorflow_datasets as tfds
import grain.python as grain

from universe import Universe3D #local
import utils #local
import visualize #local

utils.jax_status()

#parameters
epochs = 35
num_classes = 10
layers = 8
batch_size = 100
length = 32
channels = 3
kernel_size = (25,9,9) #channel first, depth last
depth_full = 2*layers #-(channels-1) #(k-1)/2 hyperplanes per layer, counting from last channel
# depth = (depth_full+length)//( (kernel_size[0]-1)//2 ) # propagation speed is (kernel_size-1)//2 per step, go upto first channel
depth = 4

color_layers = 1
padding = color_layers*2+1
depth_input = 2*padding+channels

max_depth = depth_full + 1 + depth_input
Nx = max_depth + 2*(kernel_size[0]-1) + 4 #channel first
Ny = length + 1*(kernel_size[1]-1) + 4
Nz = length + 1*(kernel_size[2]-1) + 4
print("N:", (Nx,Ny,Nz), "depth full:", depth_full, "Epochs:", epochs, "Kernel:", kernel_size)
print("padding:", padding, "depth_input:", depth_input, "max_depth:", max_depth)

#network IO paths
path = '/home/shakes/'
# path = 'F:/'

# ==============================================================================
# Grain Transformations (Augmentations & Preprocessing)
# ==============================================================================
class RandomHorizontalFlip(grain.RandomMapTransform):
    """Randomly flips the image horizontally with a 50% probability."""

    def random_map(self, element: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        # rng is a per-record deterministic random number generator provided by Grain
        if rng.random() > 0.5:
            # element['image'] is expected to be [H, W, C]
            element['image'] = np.fliplr(element['image'])
        return element

class RandomCrop(grain.RandomMapTransform):
    """Pads the image and extracts a random crop of the original size."""

    def __init__(self, padding: int = 4):
        self.padding = padding

    def random_map(self, element: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        image = element['image']
        h, w, c = image.shape

        # Pad spatially (H, W), leave channels (C) alone
        padded_image = np.pad(
            image,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='reflect'
        )

        # Select random origin for crop
        y = rng.integers(0, 2 * self.padding + 1)
        x = rng.integers(0, 2 * self.padding + 1)

        # Crop back to original dimensions
        element['image'] = padded_image[y : y + h, x : x + w]
        return element

class Normalize(grain.MapTransform):
    """Standardizes the image to [0, 1] then applies CIFAR-10 mean/std."""

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        # Convert uint8 to float32
        image = element['image'].astype(np.float32) / 255.0

        # Standard CIFAR-10 normalization metrics
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

        image = (image - mean) / std

        return {
            'image': image,
            'label': element['label']
        }

# placing of the image channels transform
class SpreadChannelsTransform(grain.MapTransform):
    """
    A Grain-compatible transform that spreads 3 color channels
    into a 3D volume with equal padding along the channel dimension.
    """
    def __init__(self, padding_size: int = 2):
        self.padding_size = padding_size

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        image = element['image']

        # TFDS/Grain images are typically (H, W, C)
        if len(image.shape) != 3 or image.shape[-1] != 3:
            raise ValueError(f"Input image must have exactly 3 channels (H, W, C). Got {image.shape[-1]}.")

        if self.padding_size == 0:
            return element

        h, w, c = image.shape

        # 1. Create a block of zeros to insert between channels.
        zeros = np.zeros((h, w, self.padding_size), dtype=image.dtype)

        # 2. Split the input image into its 3 individual channels.
        # Slicing keeps the spatial dimensions intact: (H, W, 1)
        c0 = image[:, :, 0:1]
        c1 = image[:, :, 1:2]
        c2 = image[:, :, 2:3]

        # 3. Concatenate the channels with the zero blocks interleaved in between
        # Concatenating along the channel axis (-1)
        element['image'] = np.concatenate((c0, zeros, c1, zeros, c2), axis=-1)

        return element

# ==============================================================================
# 2. DataLoader Creation Factory
# ==============================================================================
def create_cifar10_dataloader(
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1, # Use None for infinite repeating
    worker_count: int = 4,
    drop_remainder: bool = True,
    data_dir: str = None,
    download: bool = True
) -> Tuple[grain.DataLoader, int]:
    """
    Creates a Grain DataLoader for the CIFAR-10 dataset.

    Returns:
        A tuple of (DataLoader, total_images_in_split)
    """
    # 1. Download and prepare data source
    builder = tfds.builder("cifar10", data_dir=data_dir, file_format="array_record")

    if download:
        builder.download_and_prepare()

    # Get total images dynamically for the requested split
    total_images = builder.info.splits[split].num_examples

    # Use tfds.data_source to read TFDS records efficiently without tf.data
    data_source = tfds.data_source("cifar10", split=split, data_dir=data_dir)

    # 2. Build the processing pipeline
    operations = []

    # Add data augmentations specifically for the training split
    if split == "train":
        operations.extend([
            RandomCrop(padding=4),
            RandomHorizontalFlip()
        ])

    # Normalize and Batch (applies to all splits)
    operations.extend([
        Normalize(),
        SpreadChannelsTransform(padding_size=padding),
        grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ])

    # 3. Handle device sharding (Multi-GPU/TPU setup for JAX)
    # This ensures each JAX process (e.g., node) gets a distinct slice of the dataset.
    shard_options = grain.ShardOptions(
        shard_index=jax.process_index(),
        shard_count=jax.process_count(),
        drop_remainder=drop_remainder
    )

    # 4. Create the Sampler
    # IndexSampler decides the iteration order (shuffling/repeating) upfront.
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed
    )

    # 5. Instantiate the Grain DataLoader
    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count
    )

    return dataloader, total_images

# Create the dataloader and get total images
train_loader, total_images = create_cifar10_dataloader(
    split="train",
    batch_size=batch_size,
    shuffle=True,
    num_epochs=epochs+1, #only upto this
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

test_loader, total_test_images = create_cifar10_dataloader(
    split="test",
    batch_size=batch_size,
    shuffle=False,
    num_epochs=epochs+1, #only upto this
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

total_batches = total_images // batch_size
total_test_batches = total_test_images // batch_size

# Grain dataloaders yield standard python dictionaries with numpy arrays.
# For JAX, we iterate through the loader and move the arrays to the device.
train_ds = iter(train_loader)
test_ds = iter(test_loader)

#locations
#MLP with 784 inputs and 10 outputs, 3 layers inbetween
#setup network position
centerx = Nx//2
centery = Ny//2
centerz = Nz//2
#input
input_top = centerx-max_depth//2
input_left = centery-length//2
input_inner = centerz-length//2
#output, doesn't have to be at end of layers, propagator will learn directions to output
output_top = input_top+max_depth
output_left = centery
output_inner = centerz-num_classes//2 #include out layer
# print("input_top", input_top, "input_left", input_left, "output_top", output_top, "output_left", output_left)
#setup parameters and lengths
input_size = (depth_input,length,length)
input_offset = (input_top,input_left,input_inner)
output_size = (1,1,num_classes)
output_offset = (output_top,output_left,output_inner)
activation = jax.nn.leaky_relu
final_activation = jax.nn.leaky_relu
directions = 8 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "directions:", directions, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D((Nx,Ny,Nz), sharpen=False)
    mlp.initialize_3D(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Make the network and optimizer. Haiku standard
total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# AGC should typically be applied BEFORE the optimizer (SGD/Adam) updates are calculated
# or AFTER, depending on interpretation. In NFNets, it clips the raw gradients.
# clip_factor = 0.01 #AGC parameter
# opt = optax.chain(
#     optax.adamw(learning_rate=schedule),
#     utils.adaptive_gradient_clipping(clipping_threshold=clip_factor),
# )

# Initialize network and optimiser; note we draw an input to get shapes.
jax_batch = jax.tree.map(jnp.asarray, next(train_ds))
x = jax_batch['image']
# x = jnp.zeros((batch_size,length,length,channels))
# print("INIT SHAPE", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)

# plot first batch
center_point = ((depth_input-1)//2, length//2, length//2)
visualize.plot_orthogonal_views(x[0], center=center_point, cmap="gray")

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total number of parameters: {param_count}")

center_point = (output_top, centery, centerz)
# plt.imshow(params['universe3_d']['s'][:,center,:])
# plt.title("Initial States")
# plt.savefig("vnn_3D_classify_cifar_states_init.png", dpi=150)
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
    logits = logits.reshape(logits.shape[0],num_classes)
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

    # visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=input_offset)
    # visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=output_offset)
    # break

    if epoch % 1 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for test_step in trange(total_test_batches):
            # Convert numpy arrays to JAX DeviceArrays
            jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

            x = jax_batch['image']
            y = jax_batch['label']

            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
            predicted = logits.reshape(logits.shape[0],num_classes)
            predicted = jnp.argmax(predicted, axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Epoch {epoch}, Loss {total_loss}, Test Accuracy {accuracy}]")
    else:
        print(f"[Epoch {epoch}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

visualize.plot_orthogonal_views(state['universe3_d']['universe'], center=output_offset)

jnp.savez("3D_classify_cifar_cells.npz",
            weights=params['universe3_d']['w'],
            biases=params['universe3_d']['b'],
            filters=params['universe3_d']['f'],
            states=params['universe3_d']['s'],
            N=(Nx,Ny,Nz), length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

#test performance
correct = 0
total = 0
for test_step in trange(total_test_batches):
    # Convert numpy arrays to JAX DeviceArrays
    jax_batch = jax.tree.map(jnp.asarray, next(test_ds))

    x = jax_batch['image']
    y = jax_batch['label']

    logits, state_test = network.apply(params, state, x)
    predicted = logits.reshape(logits.shape[0],num_classes)
    predicted = jnp.argmax(predicted, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

plt.imshow(params['universe3_d']['w'][:,centery,:])
plt.title("Final Weights")
plt.savefig("vnn_3D_classify_cifar_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe3_d']['b'][:,centery,:])
plt.title("Final Biases")
plt.savefig("vnn_3D_classify_cifar_biases.png", dpi=150)
plt.show()

center_point = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
# plt.imshow(params['universe3_d']['f'][0,:,center,:,0])
# plt.title("Final Filter")
# plt.savefig("vnn_3D_classify_cifar_filter.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['f'][0,...,0], center=center_point)
visualize.plot_orthogonal_views(params['universe3_d']['f'][1,...,0], center=center_point)

plt.imshow(params['universe3_d']['s'][:,centery,:])
plt.title("Final States")
plt.savefig("vnn_3D_classify_cifar_states.png", dpi=150)
plt.show()

center_point = (input_top, centery, centerz)
# plt.imshow(params['universe3_d']['s'][:,centery,:])
# plt.title("Final States")
# plt.savefig("vnn_3D_classify_cifar_states.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=center_point)

states = params['universe3_d']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states[:,centery,:])
plt.title("Final States (Thresholded)")
plt.savefig("vnn_3D_classify_cifar_states_thres.png", dpi=150)
plt.show()

plt.imshow(state['universe3_d']['universe'][:,centery,:])
plt.title("Final Universe")
plt.savefig("vnn_3D_classify_cifar_universe.png", dpi=150)
plt.show()

print("END")
