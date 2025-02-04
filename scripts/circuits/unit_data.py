'''
Generate binary sequences
'''
import numpy as np

n = 8 #bits
bit_depth = 2**(n-1)
samples = 16
print("bit depth", bit_depth)

#generate random nums
numbers = np.random.randint(1, bit_depth, size=(samples,1), dtype=np.uint8)
operands = np.random.randint(1, bit_depth, size=(samples,1), dtype=np.uint8)
results = numbers + operands
print(numbers)
print(operands)
print(results)

#to binary
# for x1, x2, y in zip(numbers, operands, results):
#     bin1 = np.binary_repr(x1)
#     bin2 = np.binary_repr(x2)
#     biny = np.binary_repr(y)
print(np.unpackbits(numbers, axis=1, count=n))
