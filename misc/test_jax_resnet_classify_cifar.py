'''
ResNet for classifying CIFAR10 dataset
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

# Make the network and optimizer. Haiku standard
total_steps = (total_batches*epochs)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.02*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
x, y = next(iter(train_loader))
# x = jnp.asarray(x.numpy())
x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1], is_training=True)
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
for step in range(epochs):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for x, y in tqdm(train_loader):
        # x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())

        x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
        # print("TRAIN SHAPE", x.shape)

        params, opt_state, loss_value, state = update(params, opt_state, state, x, y)
        total_loss += loss_value/batch_size
        # break
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

            logits, state_test = network.apply(params, state, x, is_training=False) #if reshaped, use squeeze
            # predicted = jnp.squeeze(logits, axis=-1)
            predicted = jnp.argmax(logits, axis=-1)
            correct += jnp.sum(predicted == y)
            total += y.shape[0]
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {total_loss}, Test Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

#test performance
correct = 0
total = 0
for x, y in tqdm(test_loader):
    # x = jnp.asarray(x.numpy())
    y = jnp.asarray(y.numpy())

    x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last

    logits, state_test = network.apply(params, state, x, is_training=False)
    # predicted = jnp.squeeze(logits, axis=-1)
    predicted = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(predicted == y)
    total += y.shape[0]

accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

print("END")
