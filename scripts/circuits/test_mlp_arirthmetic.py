'''
Train MLP for XOR problem
Improvements from: https://github.com/VCHui/xor_pytorch
'''
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import FloatTensor

from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
epochs = 2000
n = 8
bit_depth = 2**(n-1)
samples = 6400
batch_size = 160
num_batches = samples//batch_size
delta = 0.1

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
# X_preproc = np.concatenate([X1, Xsep, X2], axis=1)
X_preproc = np.concatenate([X1, X2], axis=2)
print(X_preproc.shape, X_preproc)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
print("X", X[:3])
print("Y", Y[:3])

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)

#model
class XORNet(nn.Module):
    """A classical n-bit adder neural network
    """

    def __init__(self):
        super(XORNet, self).__init__()
        self.fc0 = nn.Linear(2*n,4*n)
        self.fc1 = nn.Linear(4*n,2*n)
        self.fc2 = nn.Linear(2*n,n)
        # self.fc3 = nn.Linear(4,n)

    def forward(self,x):
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        return F.sigmoid(self.fc2(x))

    def setparams_zeros(self):
        for p in self.parameters():
            p.data.zero_()

    def setparams_uniforms(self,delta=1):
        for p in self.parameters():
            p.data.uniform_(-delta,+delta)

model = XORNet()
model = model.to(device)

loss = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.01)

#training
model.train()
for epoch in range(epochs):
    epsilonsum = 0
    for x, y in zip(X, Y):
        # print(x.shape, y.shape)
        y_hat = model(FloatTensor(x).to(device))
        y = FloatTensor(y).to(device)
        optim.zero_grad()
        epsilon = loss(y,y_hat)
        epsilonsum += epsilon.item()
        epsilon.backward()
        optim.step()
    if epoch % 10 == 0:
        print('{:<8} {:.4e}'.format(epoch,epsilonsum/batch_size))

#testing
model.eval()
Y_pred = []
for x, y in zip(X_test, Y_test):
    y_hat = model(FloatTensor(x).to(device))
    y_hat = torch.round(y_hat).cpu().detach().numpy().astype(np.uint8)
    predictions = np.packbits(y_hat,axis=1)
    Y_pred.append(predictions)

    x1 = np.round(x[0,:n]).astype(np.uint8)
    x2 = np.round(x[0,n:]).astype(np.uint8)
    y = y[0].astype(np.uint8)
    print(np.packbits(x1,axis=0),np.packbits(x2,axis=0), predictions[0], np.packbits(y,axis=0))
Y_pred = np.array(Y_pred)
# print(Y_pred.shape, Y_packed.shape)
accuracy = np.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")
