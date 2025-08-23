'''
Utility functions for library
'''
import jax
import jax.numpy as jnp
import haiku as hk
import imageio as iio
from functools import partial
from jax import vmap

def jax_status():
    print("Total Devices:",jax.devices())
    print("Used Devices:",jax.local_devices())
    print("Using", jnp.ones(3).device.device_kind)

# required to enable hk.Initializer to be pure func and accessed as func
# see https://github.com/google-deepmind/dm-haiku/issues/24
def initializer_to_function(cls):
    def sample(shape, dtype):
        return cls(shape, dtype)
    return partial(hk.transform(sample, apply_rng=True).apply, {})

def load_sprite(filename):
    '''
    Loads sprite image as cells of different states
    Assumes that inputs are < 100, neurons < 200 and outputs < 255.
    Returns size and the list of corrdinates of the inputs, neurons and outputs
    '''
    model_sprite = iio.imread(filename)
    N = max(model_sprite.shape[0], model_sprite.shape[1])
    # print("N:", N)

    inputs = []
    neurons = []   
    outputs = []
    for i, row in enumerate(model_sprite):
        for j, col in enumerate(row):
            point = [i,j]
            if col == 0:
                continue
            elif col < 100:
                inputs.append(point)
            elif col < 200:
                neurons.append(point)
            else: #col < 255
                outputs.append(point)
    
    return N, inputs, neurons, outputs

def scatter(input, dim, index, src, reduce=None):
    # JAX-port of PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
    # Writes all values from the tensor src into input at the indices specified in the index tensor. 
    # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
    # One can simplify this implementation with ndarray.at (ref: https://github.com/google/jax/issues/8487)

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    
    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul
        
    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)

    for _ in range(len(input.shape)-1):
        _scatter = vmap_inner(_scatter)
    swap = lambda x: jnp.swapaxes(x, dim, -1)
    input, index, src = list(map(swap, (input, index, src)))
    return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))
