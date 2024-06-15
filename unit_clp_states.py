'''
Unit test for generating circular MLPs states
'''
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt

N = 17
P = 8
radius = 9
center = (8,4)
tolerance = 0.3

states = jnp.zeros((N,N))

coords = []
for x, row in enumerate(states):
    for y, col in enumerate(row):
        xDash = x - center[0]
        yDash = y - center[1]
        r = math.sqrt(xDash**2 + yDash**2)
        if r < radius + tolerance and r > radius - tolerance:
            states = states.at[x,y].set(1)
            point = [xDash, yDash]
            coords.append(point)

coords = jnp.array(coords)
print("Points:", coords.shape, coords)

# extent = (0, states.shape[1], states.shape[0], 0)
# plt.imshow(states, extent=extent)
plt.pcolormesh(states, edgecolors='k', linewidth=2)
plt.title("CLP States")
ax = plt.gca()
ax.set_aspect('equal')
# plt.grid(color='w', linewidth=1)
# plt.savefig("CLP_States.png", dpi=150)
plt.show()
