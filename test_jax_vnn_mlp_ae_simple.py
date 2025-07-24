'''
Von Neumann network that mimics an MLP for autoencoding
'''
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from universe import UniverseLP #local
from data import generate_square_pulses
import utils #local

utils.jax_status()

#parameters
epochs = 25
length = 16
pulse_width = 4
amp = 1.0
layers = 4
batch_size = 64
samples = 12800
total_batches = samples//batch_size
N = length*8
print("N:", N, "Epochs:", epochs)

#===========
#transform preproc
X = []
for index in range(total_batches):
    X.append(generate_square_pulses(
            length=length,
            pulse_width=pulse_width,
            amplitude=amp,
            batch_size=batch_size,
            z_score_standardize=False
        ))
X = jnp.array(X)
print("X Shape:", X.shape)

#locations
#MLP with 784 inputs and 10 outputs, 3 layers inbetween
#setup network position
center = N//2
depth = layers*2+1 #1 col of signal cells per layer
input_top = center-length//2
input_left = center-(depth-1)//2
output_top = center-length//2
output_left = input_left+depth+1 #include out layer
#setup parameters and lengths
input_size = (length,1)
input_offset = (input_top,input_left)
output_size = (length,1)
output_offset = (output_top,output_left)
kernel_size = (17,3)
activation = jax.nn.tanh
final_activation = None
print("depth:", depth, "total batches:", total_batches)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N)
    mlp.initialize_mlp(input_size, output_size, input_offset, output_offset, depth, kernel_size)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
# dataset_iter = iter(train_loader)
# x, y = next(dataset_iter)
# x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
# x = jnp.reshape(x, (x.shape[0], length, 1))
# print("INIT SHAPE", x[0:1].shape)
x = X[0][:, :, jnp.newaxis] #fake 2D
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)

plt.imshow(params['universe_lp']['s'])
plt.title("Initial States")
plt.savefig("vnn_mlp_full_ae_states_init.png", dpi=75)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, epsilon=1e-8) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    logits, state = network.apply(params, state, batch)
    # ce = -1 * jnp.mean(labels * jnp.log(logits + epsilon) + (1 - labels) * jnp.log(1. - logits + epsilon))

    # return ce, state
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
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(epochs+1):
    total_loss = 0
    # Do SGD on a batch of training examples.
    for x in X:
        x = x[:, :, jnp.newaxis] #fake 2D
        params, opt_state, loss_value, state = update(params, opt_state, state, x, x)
        total_loss += loss_value/batch_size
    total_loss /= total_batches

    if step % 5 == 0:
        # Periodically evaluate on train & test sets.
        correct = 0
        total = 0
        for x in X:
            x = x[:, :, jnp.newaxis] #fake 2D
            logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze
            correct += optax.squared_error(logits, x).mean()
            total += x.shape[0]
        accuracy = correct / total
        print(f"[Step {step}, Loss {total_loss}, Train MSE {accuracy}]")
    else:
        print(f"[Step {step}, Loss {total_loss}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("mlp_classify_mnist_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=params['universe_lp']['s'], 
            N=N, length=length, pulse_width=pulse_width, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_mlp_ae_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_mlp_ae_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_mlp_ae_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_mlp_full_ae_states.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_mlp_full_ae_states_thres.png", dpi=75)
plt.show()

# --- Plotting the results for visualization ---
# inference
x = X[0][:, :, jnp.newaxis] #fake 2D
logits, state_test = network.apply(params, state, x) #if reshaped, use squeeze

# Create a figure to display the first few examples from the batch
plot_size = 4
fig, axes = plt.subplots(
    plot_size, 1, 
    figsize=(6, 12), 
    sharex=True, 
    sharey=True
)
fig.suptitle('1D Square Pulse Reconstructions', fontsize=16)

for i, ax in enumerate(axes):
    ax.plot(logits[i], 'r-')
    ax.plot(x[i], 'g-')
    ax.set_title(f"Pulse {i+1}")

# Set common labels and limits
axes[-1].set_xlabel("Sample Index")
fig.text(0.06, 0.5, 'Amplitude', va='center', rotation='vertical')
# plt.ylim(-amp * 0.1, amp * 1.1)

plt.tight_layout(rect=[0.07, 0.03, 1, 0.95])
plt.show()

print("END")