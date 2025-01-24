'''
Unit test for check init for square cells in Universe module
'''
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt

from universe import CoordinateCells

from functools import partial
# required to enable hk.Initializer to be pure func and accessed as func
# see https://github.com/google-deepmind/dm-haiku/issues/24
def initializer_to_function(cls):
    def sample(shape, dtype):
        return cls(shape, dtype)
    return partial(hk.transform(sample, apply_rng=True).apply, {})

length = 3
padding = 6
num_classes = 3
N = int(length/2+0.5)+padding

#input coords
in_coords = []
in_coords.append([1,2])
in_coords.append([3,1])
in_coords.append([5,2])

#output
out_coords = []
out_coords.append([2,6])
out_coords.append([3,6])
out_coords.append([4,6])

in_coords = jnp.array(in_coords)
out_coords = jnp.array(out_coords)

#init states (integers)
#set input and output states
s_init = initializer_to_function(CoordinateCells(in_coords, out_coords, is_states=True))
states = s_init(rng=jax.random.PRNGKey(42), shape=(N,N), dtype=jnp.float32)
print("s_init", states.shape, type(states))

plt.imshow(states)
plt.title("Initial States")
# plt.savefig("vnn_mlp_classify_mnist_weights.png", dpi=150)
plt.show()
