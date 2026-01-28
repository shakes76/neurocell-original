'''
Von Neumann network that mimics an MLP with a Turing machine structure.

We use cells like a tape
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
import time, math

from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import data
import utils #local

utils.jax_status()

#parameters
epochs = 200
n = 8
bit_depth = 2**(n-1)
ctrl_bits = 4
ctrl_bit_depth = 2**ctrl_bits
samples = 64000
batch_size = 128
kernel_size = (23,13)
num_batches = samples//batch_size
total_steps = num_batches*epochs
delta = 0.2
padding = 3*max(kernel_size[0], kernel_size[1]) #on one side
epsilon = 1e-8
start_state = 1 #set None for random state
N = 2*n + padding
print("N:", N, "Epochs:", epochs, "bit depth:", bit_depth)

#===========
#data
X1f, X2f, Cf, Y_raw = data.generate_arithmetic_logic(samples, n, ctrl_bits, batch_size, delta)

#stack
X_preproc = np.concatenate([X1f, X2f, Cf], axis=-1)
print("X_preproc:", X_preproc.shape)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
# print("X", X[:1])
# print("Y", Y[:1])

#split back to separate inputs/ctrl after train/test splits
X_splits = np.dsplit(X, 3)
print("X Split:", X_splits[0].shape)
X1 = X_splits[0]
X2 = X_splits[1]
C = X_splits[2]

X_test_splits = np.dsplit(X_test, 3)
# print("X Test Split:", X_test_splits[0].shape)
X1_test = X_test_splits[0]
X2_test = X_test_splits[1]
C_test = X_test_splits[2]

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
# print("Y_packed", Y_packed[:1])

#locations
#MLP with 13 inputs and 3 outputs, 2 layers inbetween
#setup network position, but in a line like a tape
#inputs and outputs sequentially in 1D with gap
center = N//2
gap = 1
ctrl_gap = 1
input_top = center #centre vertically
input_left = center-(4*n+ctrl_gap+gap)//2 #mid of cells by half of 2*n
output_mid = center #centre vertically
output_left = input_left+(3*n+gap+ctrl_gap) #include out layer
#setup parameters and lengths
input_size = (1,n) #column vec
input_offset_x1 = (input_top,input_left)
input_offset_x2 = (input_top,input_left+n)
ctrl_size = (1,n) #row vec
ctrl_offset = (input_top,input_left+2*n+ctrl_gap)
output_size = (1,n) #column vec
output_offset = (output_mid,output_left)
activation = jax.nn.tanh
final_activation = None
directions = 16 #how many filters, i.e. possible directions, to learn
# stride = int(min(kernel_size[0], kernel_size[1])/2.+0.5) #rnd
# depth = 2*int((output_left-input_left)/stride + 0.5) #rnd, up and down vertically, hence x2
depth = 6
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N, sharpen=False)
    mlp.initialize([input_size, input_size, ctrl_size], output_size, [input_offset_x1, input_offset_x2, ctrl_offset], output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.001, peak_value=0.03, warmup_steps=int(0.025*total_steps), decay_steps=total_steps)
opt = optax.adamw(lr_schedule)
# opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = [X1[0], X2[0], C[0]] #one batch
for index, input in enumerate(x):
    x[index] = x[index][:, :, jnp.newaxis] #fake 2D
print("X0 shape:", x[0].shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

plt.imshow(params['universe_lp']['s'])
plt.title("Initial States")
plt.savefig("vnn_alu_tm_arithmetic_states_init.png", dpi=300)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    MSE
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=1) #remove extra dim, must match output vector type

    return optax.squared_error(predictions=logits, targets=labels).mean(), state

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        state,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x1, x2, c, y in zip(X1, X2, C, Y):
        # print(x.shape)
        x = [x1, x2, c] #batch of inputs
        for index, input in enumerate(x):
            x[index] = x[index][:, :, jnp.newaxis] #fake 2D
        # print("X1 shape:", x[0].shape)
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size

        # x1 = jnp.round(x[0,:n,0]).astype(np.uint8)
        # x2 = jnp.round(x[0,n:,0]).astype(np.uint8)
        # y = y[0].astype(np.uint8)
        # print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), jnp.packbits(y,axis=0))
        # break

    if step % 10 == 0:
        Y_pred = []
        done = 0
        for x1, x2, c, y in zip(X1, X2, C, Y):
            x = [x1, x2, c] #batch of inputs
            for index, input in enumerate(x):
                x[index] = x[index][:, :, jnp.newaxis] #fake 2D
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.squeeze(logits, axis=1) #remove extra dim, must match output vector type
            predictions = jnp.round(predictions).astype(np.uint8)
            predictions = jnp.packbits(predictions,axis=1)
            Y_pred.append(predictions)
            if done < 10:
                x1 = jnp.round(x1[0,:n]).astype(np.uint8)
                x2 = jnp.round(x2[0,:n]).astype(np.uint8)
                c = jnp.round(c[0,:n]).astype(np.uint8)
                y = y[0].astype(np.uint8)
                print(jnp.packbits(x1,axis=0), jnp.packbits(x2,axis=0), jnp.packbits(c,axis=0), predictions[0], jnp.packbits(y,axis=0))
                done += 1
        # Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        Y_pred = jnp.array(Y_pred)
        # print(Y_pred.shape, Y_packed.shape)
        accuracy = jnp.mean(Y_pred == Y_packed)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

jnp.savez("alu_tm_arithmetic_cells.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, ctrl_bits=ctrl_bits, bit_depth=bit_depth, n=n, input_size=input_size, output_size=output_size,
            input_offset_x1=input_offset_x1, input_offset_x2=input_offset_x2,
            crtl_size=ctrl_size, ctrl_offset=ctrl_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_alu_tm_arithmetic_weights.png", dpi=300)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_alu_tm_arithmetic_biases.png", dpi=300)
plt.show()

fig_width = int(math.sqrt(directions))
fig = plt.figure(figsize=(2*fig_width,2*fig_width))
for c1 in range(fig_width):
    for c2 in range(fig_width):
        c = c1*fig_width+c2
        plt.subplot(fig_width, fig_width, c+1)
        plt.imshow(params['universe_lp']['f'][0,:,:,c])
        plt.tight_layout()
        plt.axis('off') # Hides both x and y axes
        plt.title("Filter "+str(c))
plt.savefig("vnn_alu_tm_arithmetic_filter.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_alu_tm_arithmetic_states_final.png", dpi=300)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_alu_tm_arithmetic_states_final_thres.png", dpi=300)
plt.show()

#test performance
Y_pred = []
for x1, x2, c, y in zip(X1_test, X2_test, C_test, Y_test):
    x = [x1, x2, c] #batch of inputs
    for index, input in enumerate(x):
        x[index] = x[index][:, :, jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.squeeze(logits, axis=1) #remove extra dim, must match output vector type
    predictions = jnp.round(predictions).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")
