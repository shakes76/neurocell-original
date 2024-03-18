'''
Test the Universe object
'''
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt

from universe import Universe

#parameters
N = 32
length = 8
num_classes = 1

#===========
input_size = (length,1)
inputs = jnp.ones(input_size)
input_offset = (8,4)
# output_size = (1,1)
# output_offset = (12, 8)
output_size = input_size
output_offset = input_offset

#===========
##create model
def _forward(batch, input_offset, output_size, output_offset) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe(N)
    return mlp(batch, input_offset, output_size, output_offset)

# Make the network. Haiku standard
network = hk.without_apply_rng(hk.transform(_forward))

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), inputs, input_offset, output_size, output_offset)

plt.imshow(params['universe']['w'])
plt.title("Init")
plt.show()

outputs = network.apply(params, inputs, input_offset, output_size, output_offset)
print(outputs.shape, outputs)
# print(params['universe']['w'].dtype, params['universe']['w'].shape)
# print(params['universe']['w'])
# print(params)

# plt.imshow(params['universe']['w'])
plt.imshow(outputs)
plt.title("Output")
plt.show()
