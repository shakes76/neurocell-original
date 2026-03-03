'''
ResNet for classifying CIFAR10 dataset
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
import tensorflow_datasets as tfds
import grain.python as grain

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
    num_epochs=epochs,
    worker_count=0, # Use 0 for debugging in the main thread
    data_dir=path+'data'
)

test_loader, total_test_images = create_cifar10_dataloader(
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
class ResNetBlock(hk.Module):
    '''
    Residual Net Block
    Supports dilation via rate that results in a dilated residual net (DRN) block
    '''
    def __init__(self, channels, stride, rate=1, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.proj_conv = hk.Conv2D(
            output_channels=channels,
            kernel_shape=1,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="shortcut_conv")

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.proj_batchnorm = hk.BatchNorm(name=self.name+"_shortcut_bn", **self.bn_config)

        conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_0")
        bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=1,
            rate=rate,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_1")
        bn_1 = hk.BatchNorm(name=self.name+"_bn_1", **self.bn_config)

        self.layers = ((conv_0, bn_0), (conv_1, bn_1))

    def __call__(self, x, is_training, test_local_stats=False):
        out = shortcut = x

        shortcut = self.proj_conv(shortcut)
        shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)

class ResNet(hk.Module):
    '''
    A standard residual network (ResNet 18)
    Assumes image is input and returns logits
    conv_layers defines this many conv layers
    channels sets the initial number of filters
    Subsequent conv layers are increased by powers of 2
    dense_size sets the largest dense layer size right after conv layers
    and uses three dense layers down (each downsampled by 2) to output_size
    '''
    def __init__(self, output_size, channels=64, bn_config=None, initial_conv_config=None, logits_config=None, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.channels = channels
        self.bn_config = dict(bn_config or {})

        #batch norm config
        self.bn_config.setdefault("decay_rate", 0.9)
        self.bn_config.setdefault("eps", 1e-5)
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)

        #initial layer configs
        self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm", **self.bn_config)

        self.initial_conv_config = dict(initial_conv_config or {})
        self.initial_conv_config.setdefault("output_channels", self.channels)
        self.initial_conv_config.setdefault("kernel_shape", 3)
        self.initial_conv_config.setdefault("stride", 1)
        self.initial_conv_config.setdefault("with_bias", False)
        self.initial_conv_config.setdefault("padding", "SAME")
        self.initial_conv_config.setdefault("name", "initial_conv")

        self.initial_conv = hk.Conv2D(**self.initial_conv_config)

        self.logits_config = dict(logits_config or {})
        self.logits_config.setdefault("w_init", jax.numpy.zeros)
        self.logits_config.setdefault("name", "logits")

    def __call__(self, x, is_training, test_local_stats=False):
        '''
        ConvNet Network
        '''
        #define conv layers
        #initial block
        net=self.initial_conv(x)
        net=self.initial_batchnorm(net, is_training, test_local_stats)
        net=jax.nn.relu(net)
        #Res Blocks
        #res layer 1
        net=ResNetBlock(self.channels, stride=1, bn_config=self.bn_config, name="resblock_1a")(net, is_training)
        net=ResNetBlock(self.channels, stride=1, bn_config=self.bn_config, name="resblock_1b")(net, is_training)
        #res layer 2
        net=ResNetBlock(self.channels*2, stride=2, bn_config=self.bn_config, name="resblock_2a")(net, is_training)
        net=ResNetBlock(self.channels*2, stride=1, bn_config=self.bn_config, name="resblock_2b")(net, is_training)
        #res layer 3
        net=ResNetBlock(self.channels*4, stride=2, bn_config=self.bn_config, name="resblock_3a")(net, is_training)
        net=ResNetBlock(self.channels*4, stride=1, bn_config=self.bn_config, name="resblock_3b")(net, is_training)
        #res layer 4
        net=ResNetBlock(self.channels*4, stride=2, bn_config=self.bn_config, name="resblock_4a")(net, is_training)
        net=ResNetBlock(self.channels*4, stride=1, bn_config=self.bn_config, name="resblock_4b")(net, is_training)
        # GlobalAveragePooling2D
        net=jax.numpy.mean(net, axis=[1, 2])
        net=hk.Linear(self.output_size, **self.logits_config)(net)

        return net

def _forward(batch, is_training: bool) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    cnn = ResNet(num_classes, channels=64) #less downsampling in initial block than ResNet18
    return cnn(batch, is_training=is_training)

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
