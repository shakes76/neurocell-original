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

from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
epochs = 400
n = 8
bit_depth = 2**(n-1)
control_bits = 4
length = 2*n
samples = 320
batch_size = 16
kernel_size = (13,13)
num_batches = samples//batch_size
delta = 0.2
padding = 2*max(kernel_size[0], kernel_size[1]) #on one side
epsilon = 1e-8
start_state = 1 #set None for random state
N = length + 2*padding
print("N:", N, "Epochs:", epochs, "batch_size:", batch_size, "bit depth:", bit_depth)

#===========
##data
#   S_2     S_1     S_0     C_in    |   Out        C_out
#   0       0       0       0       |   F = A       0 Identity
#   0       0       0       1       |   F = A+1     0 Increment
#   0       0       1       0       |   F = A+B     0 Sum
#   0       0       1       1       |   F = A+B+1   1 Carry
#   0       1       1       1       |   F = A-B-1   1 Borrow
#   0       1       1       0       |   F = A-B     0 Subtract
#   0       1       0       1       |   F = A-1     0 Decrement
#   0       1       0       0       |   F = A       0 Identity
#   1       0       0       0       |   F = A+B     0 OR
#   1       0       1       0       |   F = Acirc+B 0 XOR
#   1       1       0       0       |   F = AB      0 AND
#   1       1       1       0       |   F = A'      0 NOT
#data
numbers = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)
operands = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)
controls = np.random.randint(1, 2**control_bits, size=(num_batches,batch_size,1), dtype=np.uint8)
print("numbers:", numbers.shape)

#create labels
Y_raw = []
for batch_numbers, batch_operands, batch_controls in zip(numbers, operands, controls):
    batch = []
    # print("batch numbers shape",batch_numbers.shape)
    for number, operand, control in zip(batch_numbers, batch_operands, batch_controls):
        # print("number shape",number.shape)
        # print(number)
        if control == 0 or control == 4: #Identity
            batch.append(number)
        elif control == 1: #Increment
            batch.append(number+1)
        elif control == 2 or control == 8: #Sum, AND
            batch.append(number+operand)
        elif control == 3: #Sum, Increment
            batch.append(number+operand+1)
        elif control == 5: #Decrement
            batch.append(number-1)
        elif control == 6: #Minus
            batch.append(number-operand)
        elif control == 7: #Minus, decrement
            batch.append(number-operand-1)
        elif control == 12: #Multiply
            batch.append(number*operand)
        else: #Identity
            batch.append(number)
    # print(batch)
    Y_raw.append(batch)
Y_raw = np.array(Y_raw)
print("Y_raw:", Y_raw.shape)
Y_raw = Y_raw.reshape(num_batches,batch_size,1)
print("Y_raw:", Y_raw.shape)

#to bits
X1 = np.unpackbits(numbers, axis=2, count=n)
X2 = np.unpackbits(operands, axis=2, count=n)
C = np.unpackbits(controls, axis=2, count=control_bits)
Y_raw = np.unpackbits(Y_raw, axis=2, count=n)
print("X1 shape:", X1.shape)

#add noise
rand1 = np.random.uniform(-delta,+delta,size=X1.shape)
rand2 = np.random.uniform(-delta,+delta,size=X2.shape)
rand3 = np.random.uniform(-delta,+delta,size=C.shape)
X1 = X1.astype(np.float32) + rand1
X2 = X2.astype(np.float32) + rand2
C = C.astype(np.float32) + rand3

#stack
X_preproc = np.concatenate([X1, X2, C], axis=-1)
print("X_preproc:", X_preproc.shape)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
print("X", X[:1])
print("Y", Y[:1])

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
print("Y_packed", Y_packed[:1])

print("END")