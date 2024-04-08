'''
Unit test for effect of filters
'''
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import universe #local

#parameters
N = 7
samples = 120
length = 1
num_classes = 1
lr = 0.1

#===========
##data
X = np.arange(-30, 30, 0.5, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -10) & (X[:,0] < 10) ] = 1.0

#helpers

#===========
#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
key = jax.random.key(42)
depth = output_offset[1]-input_offset[1]
print("depth:", depth)

##create model
model = universe.Universe(N)

#random init
# Initialize model
params = model.random_init()
result_init = model.get_cells()

#loss function
def loss(logits, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    return -1 * jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1. - logits))

#get output
model.set_input_vector(X[0].reshape(-1,1), input_offset)
model.set_output_vector(Y[0].reshape(-1,1), output_offset)
# result = model.get_cells()
model.forward_propagate(activate=jax.nn.sigmoid)
output = model.get_output_vector(output_size, output_offset)
result = model.get_cells() #for plot

#compute loss
loss_value = loss(output, Y[0])
step = lr*loss_value #take negative step
print(output, "Y:", Y[0], "loss:", loss_value, "step:", step)
'''
#write step and backprop to get updates to weights required
backprop = jax.lax.dynamic_update_slice(result, step, output_offset)
backprop = clear_cells(backprop, output_offset, after=True)

for i in range(depth):
    backprop = backward_propagate(backprop)

#apply corrections
universe_update = universe - backprop
#restore input (data consistency)
universe_update = jax.lax.dynamic_update_slice(universe_update, X[0].reshape(-1,1), input_offset)

#forward prop
result2 = forward_propagate(universe_update)
for i in range(depth-1):
    result2 = forward_propagate(result2)

output2 = jax.lax.dynamic_slice(result2, output_offset, output_size)
print(output2)
'''
plt.imshow(result_init)
plt.title("Initial")
plt.show()

plt.imshow(result)
plt.title("Result")
plt.show()
'''
plt.imshow(backprop)
plt.title("Backprop")
plt.show()

plt.imshow(universe_update)
plt.title("Updated")
plt.show()

plt.imshow(result2)
plt.title("Result2")
plt.show()
'''