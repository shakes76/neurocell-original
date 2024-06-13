'''
Test direct coords set to cells via JNP
'''
# import jax
import jax.numpy as jnp

import utils #local

architecture_filename = "patterns/Sprite-CLP13.png"
# dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
# dim = 0

#load network architecture
N, input_coords, neuron_coords, output_coords = utils.load_sprite(architecture_filename)
input_coords = jnp.array(input_coords)
print("N:", N)
print(input_coords.shape, input_coords)
samples = len(input_coords)

model = jnp.zeros((N,N))
inputs = jnp.ones(samples)

model = model.at[input_coords[:, 0], input_coords[:, 1]].set(2)
# model = jax.lax.scatter(model, input_coords, inputs, dnums)
# model = utils.scatter(model, dim, jnp.array(input_coords), inputs)

import matplotlib.pyplot as plt

plt.imshow(model)
plt.show()
