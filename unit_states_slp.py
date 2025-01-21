'''
Unit test for generating square MLPs states
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

N = 64
P = 8*8
center = (N//2,N//2)
offset = (8,8)
num_classes = 10

states = jnp.zeros((N,N))

# construct pattern in 3 parts: top, middle/side, and bottom
# middle is double size of top or bottom
# assumes that total length is 4n

#inputs
inputs = jnp.arange(1,P+1, dtype=jnp.float32)
#top/up
up = inputs[0:P//4].reshape(1,-1) #row vec
print("top:", up.shape, up)
up = jnp.fliplr(up) #reverse
shift = (offset[0], offset[1]+1) #one step right
states = jax.lax.dynamic_update_slice(states, up, shift)
#middle
mid = inputs[P//4+1:P//2+P//4].reshape(-1,1) #col vec
print("mid:", mid.shape, mid)
shift = (offset[0]+1, offset[1]) #one step down
states = jax.lax.dynamic_update_slice(states, mid, shift)
#bottom/down
down = inputs[P//2+P//4+1:P//2+P//4+P//4].reshape(1,-1) #row vec
print("down:", down.shape, down)
shift = tuple(map(lambda x, y: x+y, offset, mid.shape)) #move to bottom
shift = (shift[0]+1, shift[1]) #one step down
print("shift:", shift)
states = jax.lax.dynamic_update_slice(states, down, shift)

#outputs
outputs = jnp.arange(64, num_classes+64, dtype=jnp.float32)
#horizontal
# out = jnp.arange(1, num_classes+1, dtype=jnp.float32).reshape(1,-1) #row vec
# shift = (offset[0]+P//4, offset[1]+P//8) #shift by half of mid and top
# states = jax.lax.dynamic_update_slice(states, out, shift)
#vertical
out = outputs.reshape(-1,1) #col vec
out_shift = (offset[0]+P//4-num_classes//2, offset[1]+P//4+1) #shift by half of mid and top
states = jax.lax.dynamic_update_slice(states, out, out_shift)

print("depth:", out_shift[1]-offset[1])

# extent = (0, states.shape[1], states.shape[0], 0)
# plt.imshow(states, extent=extent)
plt.pcolormesh(states, edgecolors='k', linewidth=2)
plt.title("CLP States")
ax = plt.gca()
ax.set_aspect('equal')
# plt.grid(color='w', linewidth=1)
# plt.savefig("CLP_States.png", dpi=150)
plt.show()
