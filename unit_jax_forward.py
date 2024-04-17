'''
Unit test of forward propagation of neurocells
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import universe #local

#parameters
N = 7
length = 1
num_classes = 1

#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
depth = output_offset[1]-input_offset[1]
print("depth:", depth)

#data
X = jnp.ones((1,1), dtype=jnp.float32)
Y = jnp.full((1,1), 0.5, dtype=jnp.float32)

##create model
model = universe.Universe(N)

# Initialize model
params = model.zero_init()

#set output
model.set_input_vector(X, input_offset)
model.set_output_vector(Y, output_offset)

result_init = model.get_cells() #for plot

#forward pass
model.save_state() #save weights
for i in range(depth):
    model.forward_propagate(activate=jax.nn.tanh)
output = model.get_output_vector(output_size, output_offset)
result = model.get_cells() #for plot
model.restore_state() #restore to original weights

print("Output:", output)

plt.imshow(result_init)
plt.title("Initial")
plt.show()

plt.imshow(result)
plt.title("Initial")
plt.show()
