'''
Test for XOR data generation
'''
import numpy as np

bits = 2
samples = 8
length = bits*samples

x = np.zeros((length), dtype=np.int32)
x[:length//2] = 1 #equal number of 0 and 1
np.random.shuffle(x)
print("x", x)
x = np.reshape(x, (samples, bits))
print("x reshaped", x.shape, x)

y = x[:,0] ^ x[:,1]
print("y", y.shape, y)
