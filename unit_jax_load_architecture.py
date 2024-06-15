'''
Load model architecture from Sprite image
'''
import imageio as iio
import jax.numpy as jnp

model_sprite = iio.imread("patterns/Sprite-CLP13-C.png")
N = max(model_sprite.shape[0], model_sprite.shape[1])
print("N:", N)

inputs = []
neurons = []
outputs = []
for i, row in enumerate(model_sprite):
    for j, col in enumerate(row):
        point = (i,j)
        if col == 0:
            continue
        elif col < 100:
            inputs.append(point)
        elif col < 200:
            neurons.append(point)
        else: #col < 255
            outputs.append(point)

model = jnp.zeros((N,N))

for input_coord in inputs:
    i, j = input_coord
    model = model.at[i,j].set(1)

for neuron_coord in neurons:
    i, j = neuron_coord
    model = model.at[i,j].set(2)
    
for output_coord in outputs:
    i, j = output_coord
    model = model.at[i,j].set(3)

import matplotlib.pyplot as plt

plt.imshow(model)
plt.show()
