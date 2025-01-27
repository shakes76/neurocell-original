'''
Unit test for check init for square cells in Universe module
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from universe import SquareCells
import utils

length = 28*28
padding = 8
num_classes = 10
N = length//2+padding
center = N//2
offset = (padding//2,center-length//8) #center-length//2=padding, center-top

input_size = (length,1)
input_offset = offset
output_size = (num_classes,1)
output_offset = (offset[0]+length//4-num_classes//2, offset[1]+length//4+2) #shift by half of mid and top

#init states (integers)
#set input and output states
s_init = utils.initializer_to_function(SquareCells(input_size, output_size, input_offset, output_offset, is_states=True, random_init=False))
states = s_init(rng=jax.random.PRNGKey(42), shape=(N,N), dtype=jnp.float32)
print("s_init", states.shape, type(states))

plt.imshow(states)
plt.title("Initial States")
# plt.savefig("vnn_mlp_classify_mnist_weights.png", dpi=150)
plt.show()
