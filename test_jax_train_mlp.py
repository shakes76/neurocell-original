'''
Unit test for effect of filters
'''
import jax
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from time import process_time

import universe #local
import utils #local

utils.jax_status()

#parameters
N = 7
epochs = 3
samples = 120
length = 1
num_classes = 1
epsilon = 1e-6
lr = 0.1

#===========
##data
X = np.arange(-1.5, 1.5, 0.025, dtype=np.float32).reshape(-1, 1)
Y = np.zeros(samples, dtype=np.float32)

#square wave
Y[ (X[:,0] >= -0.5) & (X[:,0] < 0.5) ] = 1.0

#===========
#setup
input_size = (length,1)
input_offset = (3,1)
output_size = (num_classes,1)
output_offset = (3,5)
layer_size = (5,1)
layer_offset = (1,3)
key = jax.random.key(42)
depth = output_offset[1]-input_offset[1]
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
print("depth:", depth)

##create model
model = universe.Universe(N)

#random init
# Initialize model
model.initialize_vertical(input_size, output_size, input_offset, output_offset)
model.set_layer_vertical(layer_size, layer_offset)

time_s = process_time()
for epoch in range(1, epochs+1):
    Y_pred = []
    for iteration, (x, y) in enumerate(zip(X, Y)):
        #set output
        model.set_input_vector(x.reshape(-1,1), input_offset)
        # model.set_output_vector(y.reshape(-1,1), output_offset)
        
        #grad
        # loss_value2, grads = model.value_and_grad(depth, y.reshape(-1,1), x.reshape(-1,1), input_offset, output_size, output_offset)
        loss_value, grads, bias_grads, weights = model.value_and_grad_update(lr, depth, y.reshape(-1,1), x.reshape(-1,1), input_offset, output_size, output_offset)
        # forward = model.get_cells() #for plot

        #forward pass to check output
        #set output
        model.set_input_vector(x.reshape(-1,1), input_offset)
        # model.set_output_vector(y.reshape(-1,1), output_offset)
        for i in range(1, depth):
            model.forward_propagate(activate=activation, norm=False)
        model.forward_propagate(activate=final_activation, norm=False)
        result = model.get_cells() #for plot
        # weights = model.get_weights() #for plot
        output2 = model.get_output_vector(output_size, output_offset)
        # restore = model.get_cells() #for plot
        Y_pred.append(output2.reshape(1))
        model.reset_state() #reset to zero
        
        if iteration%10 == 0:
            # print("i:", iteration, "X", x, "Y:", y, "Forward Out:", output, "Update Out:", output2, "loss:", loss_value, "loss2:", loss_value2)
            # print("i:", iteration, "X", x, "Y:", y, "Forward Out:", output, "Update Out:", output2, "loss:", loss_value)
            print("i:", iteration, "X", x, "Y:", y, "Update Out:", output2, "loss:", loss_value)
        # break
    print(">> Epoch:", epoch, "Loss:", loss_value)
elapsed_time = process_time() - time_s
print("Training time:", elapsed_time, "s or ", elapsed_time/60.0, "mins")

#save models
model.save("cells.npz")

plt.imshow(bias_grads)
plt.title("Bias Grads")
plt.savefig("Bias_Grads.png", dpi=150)
#plt.show()

plt.imshow(grads)
plt.title("Grads")
plt.savefig("Grads.png", dpi=150)
#plt.show()

plt.imshow(result)
plt.title("Final Result")
plt.savefig("Final_Result.png", dpi=150)
#plt.show()

plt.imshow(weights)
plt.title("Final Weights")
plt.savefig("Final_Weights.png", dpi=150)
plt.show()

#function plot
fig = plt.figure()
plt.plot(X, Y_pred)
plt.plot(X, Y, 'm')
plt.xlabel('X')
plt.savefig("Plot.png", dpi=150)
plt.show()
