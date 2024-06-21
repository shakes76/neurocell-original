'''
Von Neumann network that mimics an MLP for classifying MNIST dataset
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
N = 1024
epochs = 20
num_classes = 10
layers = 3
start_state = 1 #set None for random state
print("N:", N, "Epochs:", epochs)

#===========
##data
# Load data from https://www.openml.org/d/554
X_raw, Y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='pandas')
Y_raw = Y_raw.astype(np.int32)
length = X_raw.shape[1]
samples = X_raw.shape[0]
num_classes = 10
print("Data Shape:", X_raw.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)
print("Y raw:", Y_raw.dtype, Y_raw[:10])

#scaling
scaler = StandardScaler()
X_preproc = scaler.fit_transform(X_raw)

#shuffle
random_state = check_random_state(0)
permutation = random_state.permutation(X_preproc.shape[0])
X_preproc = X_preproc[permutation]
Y_raw = Y_raw[permutation]
X_preproc = X_preproc.reshape((X_preproc.shape[0], -1))

#split
X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

#locations
#MLP with 784 inputs and 10 outputs, 3 layers inbetween
#setup network position
center = N//2
total_layers = layers*2+3 #1 col of signal cells per layer, 1 for in + signal col and 1 for out
input_top = center-length//2
input_left = center-total_layers//2-1
output_top = center-num_classes//2
output_left = input_left+total_layers-1 #dont include out obv
print("input_top", input_top, "input_left", input_left, "output_top", output_top, "output_left", output_left)
#setup parameters and lengths
input_size = (length,1)
input_offset = (input_top,input_left)
output_size = (num_classes,1)
output_offset = (output_top,output_left)
layer_sizes = [256, 128, 32]
kernel_size = (7,3)
activation = jax.nn.tanh
final_activation = None
depth = output_offset[1]-input_offset[1]
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N)
    mlp.initialize_mlp_full(input_size, output_size, input_offset, output_offset, layer_sizes, kernel_size)
    return mlp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.001)

# Initialize network and optimiser; note we draw an input to get shapes.
params, state = network.init(jax.random.PRNGKey(42), X[0].reshape(-1,1))
opt_state = opt.init(params)
# print(params)

plt.imshow(state['universe_lp']['states'])
plt.title("Final States")
plt.savefig("vnn_mlp_full_classify_states.png", dpi=75)
plt.show()

print("END")