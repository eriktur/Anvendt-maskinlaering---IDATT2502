import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Definer inputene og NAND-outputene
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
outputs = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Visualiser dataene
for i in range(4):
    plt.scatter(inputs[i, 0].item(), inputs[i, 1].item(), c='b' if outputs[i].item() == 1 else 'r', s=100)

plt.xlabel('Input A')
plt.ylabel('Input B')
plt.title('NAND Input Data')
plt.show()

class NandOperatorModel:
    def __init__(self):
        # Definer vekter og bias (to input-vekter og en bias)
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # To vekter
        self.b = torch.tensor([[0.0]], requires_grad=True)  # En bias

    # Modellen bruker sigmoid-funksjonen for å forutsi output
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Tapfunksjonen bruker binary cross-entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

# Initialiser modell og optimizer
model = NandOperatorModel()
optimizer = torch.optim.SGD([model.W, model.b], lr=0.5)  # Øk læringsraten for raskere konvergens

# Tren modellen
epochs = 10000
for epoch in range(epochs):
    loss = model.loss(inputs, outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Print de endelige parametrene etter trening
print(f'Endelig W = {model.W.tolist()}, b = {model.b.item()}, loss = {loss.item()}')

# Generer prediksjoner fra den trente modellen
with torch.no_grad():
    predicted = model.f(inputs).numpy()

# Lag et 3D-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Lag et meshgrid for inputverdier (fra -0.5 til 1.5 for å dekke litt utenfor 0 og 1)
x_range = np.linspace(-0.5, 1.5, 100)
y_range = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x_range, y_range)

# Beregn prediksjonene for meshgrid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        inp = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
        Z[i, j] = model.f(inp).item()

# Plott planen som modellen predikerer
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Plott de faktiske datapunktene som røde prikker og prediksjonene som grønne kryss
for i in range(4):
    ax.scatter(inputs[i, 0].item(), inputs[i, 1].item(), outputs[i].item(), color='r', s=100, label='True output' if i == 0 else "")
    ax.scatter(inputs[i, 0].item(), inputs[i, 1].item(), predicted[i].item(), color='g', marker='x', s=200, label='Model prediction' if i == 0 else "")

ax.set_xlabel('Input A')
ax.set_ylabel('Input B')
ax.set_zlabel('Output / Prediction')
plt.title('3D Model of NAND Operator with Prediction Plane')
plt.legend()
plt.show()
