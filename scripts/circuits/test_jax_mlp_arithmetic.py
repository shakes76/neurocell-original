'''
Von Neumann network that mimics an MLP for arithmetic
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Tuple
import time

from sklearn.model_selection import train_test_split

#parameters
epochs = 1000
n = 8
bit_depth = 2**(n-1)
length = 2*n
samples = 6400
batch_size = 160
num_batches = samples//batch_size
delta = 0.1
print("Epochs:", epochs)

#===========
#data
numbers = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)
operands = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)
results = numbers + operands
#to bits
X1 = np.unpackbits(numbers, axis=2, count=n)
X2 = np.unpackbits(operands, axis=2, count=n)
Y_raw = np.unpackbits(results, axis=2, count=n)
print(X1.shape, X1)

#add noise
rand1 = np.random.uniform(-delta,+delta,size=X1.shape)
rand2 = np.random.uniform(-delta,+delta,size=X2.shape)
X1 = X1.astype(np.float32) + rand1
# Xsep = np.zeros((samples,1), dtype=np.float32)
X2 = X2.astype(np.float32) + rand2

#stack
# X_preproc = np.concatenate([X1, Xsep, X2], axis=-1)
X_preproc = np.concatenate([X1, X2], axis=-1)
print(X_preproc.shape, X_preproc)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
print("X", X[:1])
print("Y", Y[:1])

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
# print("Y_packed", Y_packed[:1])

#===========
##create model
class MLP(hk.Module):
    '''
    Multi-layer perceptron
    Applies flattening to input and returns logits
    '''
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        '''
        MLP Network
        '''
        #define layers
        net=hk.Linear(4*n)(x)
        net=jax.nn.sigmoid(net)
        net=hk.Linear(2*n)(net)
        net=jax.nn.sigmoid(net)
        net=hk.Linear(self.output_size)(net)
        net=jax.nn.sigmoid(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(n)
    return mlp(batch)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
print("X0 shape:", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)

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
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size

    if step % 10 == 0:
        Y_pred = []
        done = 0
        for x, y in zip(X, Y):
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.round(logits).astype(np.uint8)
            predictions = jnp.packbits(predictions,axis=1)
            Y_pred.append(predictions)
            if done < 3:
                x1 = jnp.round(x[0,:n]).astype(np.uint8)
                x2 = jnp.round(x[0,n:]).astype(np.uint8)
                y = y[0].astype(np.uint8)
                print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), predictions[0], jnp.packbits(y,axis=0))
                done += 1
        Y_pred = jnp.array(Y_pred)
        accuracy = jnp.mean(Y_pred == Y_packed)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

#function plot
Y_pred = []
for x, y in zip(X, Y):
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.round(logits).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")