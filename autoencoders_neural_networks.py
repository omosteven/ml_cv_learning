import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4,2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(2,4), nn.ReLU())

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X_tensor = torch.tensor(X,dtype=torch.float32)

for epoch in range(200):
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Final loss', loss.item())
print('Model:', model)
print('Criterion', criterion)
print('Optimizer', optimizer)
print('Params:', model.parameters())
print('X Tensor', X_tensor)