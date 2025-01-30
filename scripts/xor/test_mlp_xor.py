'''
Train MLP for XOR problem
Improvements from: https://github.com/VCHui/xor_pytorch
'''
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import FloatTensor

from data import XORData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
epochs = 30
table = XORData.TRUTHTABLE
batch_size = 100
delta = 0.2

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

loss = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=0.01)

#training
model.train()
for epoch in range(epochs):
    epsilonsum = 0
    for t in XORData(batch_size,delta,table):
        # x = x.to(device)
        y = model(FloatTensor(t[0:2]).to(device))
        target = FloatTensor(t[2:]).to(device)
        optim.zero_grad()
        epsilon = loss(y,target)
        epsilonsum += epsilon.item()
        epsilon.backward()
        optim.step()
    print('{:<8} {:.4e}'.format(epoch,epsilonsum/batch_size))

#testing
model.eval()
for a,b,xor in XORData():
    y = model(FloatTensor([a,b]).to(device))
    target = FloatTensor([xor]).to(device)
    epsilon = loss(y,target)
    print('{} {:+.8f} {:+.8f}'.format(
        (int(a),int(b)),y.item(),epsilon.item()))

print("END")
