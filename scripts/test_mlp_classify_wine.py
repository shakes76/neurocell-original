'''
Simple MLP trained as a classify network for a very simple dataset - Wine
'''
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Tuple

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#parameters
iterations = 10

#===========
##data
data_dict = datasets.load_wine(return_X_y=False)
X_raw = data_dict['data']
Y_raw = data_dict['target']
length = len(data_dict['feature_names'])
samples = len(Y_raw)
num_classes = len(data_dict['target_names'])
print("Data Shape:", X_raw.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)

#scaling
scaler = StandardScaler()
X_preproc = scaler.fit_transform(X_raw)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

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
        net=hk.Flatten()(x)
        net=hk.Linear(12)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(6)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = MLP(num_classes)
    return mlp(batch)

# Make the network and optimiser.
network = hk.without_apply_rng(hk.transform(_forward))
opt = optax.sgd(learning_rate=0.01)

#loss function
def loss(params: hk.Params, batch, labels) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    logits = network.apply(params, batch)
    return -jnp.sum(labels * jax.nn.log_softmax(logits))

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value

# Initialize network and optimiser; note we draw an input to get shapes.
params = network.init(jax.random.PRNGKey(42), X)
opt_state = opt.init(params)

# Train/eval loop.
for step in range(iterations+1):
    # Do SGD on a batch of training examples.
    for x, y in zip(X, Y):
        params, opt_state, loss_value = update(params, opt_state, x, y)

    if step % 10 == 0:
        # Periodically evaluate on train & test sets.
        Y_pred = network.apply(params, X)
        accuracy = jnp.mean(jnp.argmax(Y_pred, axis=-1) == Y)
        print(f"[Step {step}, Loss {loss_value}, Train Accuracy {accuracy}]")

#test performance
Y_pred = []
for x, y in zip(X_test, Y_test):
    logits = network.apply(params, x)
    Y_pred.append(logits)
Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
accuracy = jnp.mean(Y_pred == Y_test)
print(f"[Final Accuracy {accuracy}]")

print("END")
