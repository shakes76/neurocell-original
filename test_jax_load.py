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
filename = "mlp_classify_cells.npz"
length = None
num_classes = None
input_size = None
input_offset = None
output_size = None
output_offset = None
layer_size = None
layer_offset = None
kernel_size = None

with jnp.load(filename, allow_pickle=False) as data:
    weights = data['weights']
    biases = data['biases']
    states = data['states']
    filters = data['filters']
    N=data['N']
    length=data['length']
    num_classes=data['num_classes']
    input_size=data['input_size']
    output_size=data['output_size']
    input_offset=data['input_offset']
    output_offset=data['output_offset']
    layer_size=data['layer_size']
    layer_offset=data['layer_offset']
    kernel_size=data['kernel_size']

print("N:", N, "Number of Classes:", num_classes, "kernel_size", kernel_size)
print("input_size:", input_size, "output_size:", output_size, "layer_size:", layer_size)

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
