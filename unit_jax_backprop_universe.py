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
X = np.arange(-1.5, 1.5, 0.025, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -0.5) & (X[:,0] < 0.5) ] = 1.0

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
# result_init = model.get_cells()

#loss function
def loss(logits, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    return -1 * jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1. - logits))

#set output
model.set_input_vector(X[0].reshape(-1,1), input_offset)
model.set_output_vector(Y[0].reshape(-1,1), output_offset)
result_init = model.get_cells()
model.save_state() #save weights
for i in range(depth):
    model.forward_propagate(activate=jax.nn.sigmoid)
output = model.get_output_vector(output_size, output_offset)
result = model.get_cells() #for plot
model.restore_state() #restore to original weights

#compute loss
loss_value = loss(output, Y[0])
step = lr*loss_value #take negative step
print(output, "Y:", Y[0], "loss:", loss_value, "step:", step)

#apply corrections
model.gradient_update(step.reshape(-1,1), depth, output_offset)
updated = model.get_cells() #for plot

#set output
model.set_input_vector(X[0].reshape(-1,1), input_offset)
model.set_output_vector(Y[0].reshape(-1,1), output_offset)
model.save_state() #save weights
for i in range(depth):
    model.forward_propagate(activate=jax.nn.sigmoid)
output2 = model.get_output_vector(output_size, output_offset)
print("Output:", output2)
result2 = model.get_cells() #for plot
model.restore_state() #restore to original weights

plt.imshow(result_init)
plt.title("Initial")
plt.show()

plt.imshow(result)
plt.title("Result")
plt.show()

plt.imshow(updated)
plt.title("Updated")
plt.show()

plt.imshow(result2)
plt.title("Result2")
plt.show()
