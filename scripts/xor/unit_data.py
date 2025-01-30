'''
Unit test for data module
'''
from data import XORData

table = XORData.TRUTHTABLE
batch_size = 8
delta = 0.2

for t in XORData(batch_size, delta, table):
    x = t[0:2]
    y = t[2:]
    print("t", t, "x", x, "y", y)
