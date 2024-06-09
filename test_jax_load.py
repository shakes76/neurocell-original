'''
Test load model
'''
import universe #local
import jax.numpy as jnp
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

##create model
N = 1 #unused
# model = universe.Universe(N)
filename = "mlp_classify_cells.npz"

with jnp.load(filename, allow_pickle=False) as data:
    weights = data['weights']
    biases = data['biases']
    states = data['states']
    filters = data['filters']

plt.imshow(weights)
plt.title("Final Weights")
# plt.savefig("Final_Weights.png", dpi=150)
plt.show()

plt.imshow(biases)
plt.title("Final Biases")
# plt.savefig("Final_Biases.png", dpi=150)
plt.show()

plt.imshow(states)
plt.title("Final States")
# plt.savefig("Final_States.png", dpi=150)
plt.show()

plt.imshow(filters[0,:,:,0])
plt.title("Final Filter")
# plt.savefig("vnn_mlp_classify_filter.png", dpi=75)
plt.show()
