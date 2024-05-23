'''
Test load model
'''
import universe #local
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

##create model
N = 1 #unused
model = universe.Universe(N)
filename = "cells.npz"

model.load(filename)

weights = model.get_weights() #for plot
biases = model.get_biases() #for plot
states = model.get_states() #for plot

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
