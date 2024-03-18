'''
Test coordinate assignment in JAX
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

N = 128
print("Universe Size (N):", N)
grid_size = (N,N) 

col = 4
offset = 4
length = 8
# insert_points = []
# for index in range(length):
#     insert_points.append((index,col))

input_size = (length,1)
inputs = jnp.ones(input_size)
universe = jnp.zeros(grid_size)
# universe[insert_points] = 1 #causes error in JAX
# universe = universe.at[insert_points].set(1) #causes error in JAX
universe = jax.lax.dynamic_update_slice(universe, inputs, (offset, col))

inputs_read = jax.lax.dynamic_slice(universe, (offset, col), input_size)
print(inputs_read)

plt.imshow(universe)
plt.show()
