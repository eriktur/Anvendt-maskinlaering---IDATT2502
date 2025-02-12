import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data for NOT-operatoren: input og output
inputs = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
outputs = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

# Visualiser datapunktene
plt.scatter(inputs.numpy(), outputs.numpy(), label='Data (NOT operator)')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('NOT operator data')
plt.show()

class NotOperatorModel:
    def __init__(self):
        # Vekt (W) og bias (b) for modellen, med gradientaktivering
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Modellen bruker sigmoid-funksjonen for å beregne output
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Tapfunksjon som bruker binary cross-entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

# Initialiser modell og optimaliserer
model = NotOperatorModel()
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)

# Trener modellen
epochs = 100000
for epoch in range(epochs):
    loss = model.loss(inputs, outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Print de endelige parametrene etter trening
print(f'Endelig W = {model.W.item()}, b = {model.b.item()}, loss = {loss.item()}')

# Generer prediksjoner fra den trente modellen
with torch.no_grad():
    predicted = model.f(inputs).numpy()

# Lag et 3D-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Lag et meshgrid for inputverdier (fra -1 til 2 for å dekke litt utenfor 0 og 1)
x_range = np.linspace(-1, 2, 100)
y_range = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x_range, y_range)

# Beregn prediksjonene for meshgrid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        inp = torch.tensor([[X[i, j]]], dtype=torch.float32)
        Z[i, j] = model.f(inp).item()

# Plott planen som modellen predikerer
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Plott de faktiske datapunktene som røde prikker
ax.scatter(inputs.numpy(), outputs.numpy(), outputs.numpy(), color='r', label='Data (NOT operator)', s=100)

ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.set_zlabel('Model prediction')
plt.title('3D Model of NOT Operator with Prediction Plane')
plt.legend()
plt.show()
