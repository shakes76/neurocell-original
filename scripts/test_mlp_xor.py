'''
Solves XOR learning problem
See: https://github.com/epochventures/PyTorch
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
epochs = 100
bits = 2
length = bits
samples = 160
num_classes = 1
length_data = bits*samples
delta = 0.2

#===========
##data
X_preproc = np.zeros((length_data), dtype=np.int32)
X_preproc[:length_data//2] = 1 #equal number of 0 and 1
np.random.shuffle(X_preproc)
X_preproc = np.reshape(X_preproc, (samples, bits))
Y_raw = X_preproc[:,0] ^ X_preproc[:,1] #xor
# Y_raw = X_preproc[:,0] & X_preproc[:,1] #and

#add noise (necessary)
noise = np.random.uniform(-delta,+delta,size=(samples, bits))
X_preproc = X_preproc.astype(np.float32) + noise

print("Data Shape:", X_preproc.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)
# print(X_preproc, Y_raw)
# print(X_preproc.astype(ds_dtype), Y_raw.astype(ds_dtype))
# print(X_preproc.astype(ds_dtype)[0][0], int(X_preproc.astype(ds_dtype)[0][0]))
# print(X_preproc.astype(ds_dtype)[0][1], int(X_preproc.astype(ds_dtype)[0][1]))

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

X = torch.Tensor(X)
Y = torch.Tensor(Y)
X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)
print("X shape", X.shape)

#model
class XORNet(nn.Module):
    """A classical 2-layer XOR neural network

    >>> net = XORNet()
    >>> net
    XORNet (
      (fc0): Linear (2 -> 2)
      (fc1): Linear (2 -> 1)
    )

    """

    def __init__(self):
        super(XORNet, self).__init__()
        self.fc0 = nn.Linear(2,2)
        self.fc1 = nn.Linear(2,1)

    def forward(self,x):
        x = F.sigmoid(self.fc0(x))
        return F.sigmoid(self.fc1(x))

    def setparams_zeros(self):
        for p in self.parameters():
            p.data.zero_()

    def setparams_uniforms(self,delta=1):
        for p in self.parameters():
            p.data.uniform_(-delta,+delta)

model = XORNet()
model = model.to(device)

criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#--------------
# Train the model
model.train()
print("> Training")
for epoch in range(epochs):

    for i, (x, labels) in enumerate(zip(X,Y)):
        x = x.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}] Loss: {:.5f}"
                    .format(epoch+1, epochs, i+1, loss.item()))
print("Training done") 

# Test the model
print("> Testing")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, labels in zip(X_test, Y_test):
        x = x.to(device)
        labels = labels.to(device)
        outputs = model(x)
        predicted = torch.round(outputs.data)
        total += 1
        correct += (predicted == labels).sum().item()
        print("x", x, "y", labels, "y_pred", predicted)

    print('Test Accuracy: {} %'.format(100 * correct / total))

print("Training done") 

print('END')
