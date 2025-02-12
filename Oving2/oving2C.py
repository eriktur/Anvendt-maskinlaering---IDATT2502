import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Definer inputene og XOR-outputene
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
outputs = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class XORModel:
    def __init__(self):
        # Initialiser vektene tilfeldig mellom -1 og 1
        self.W1 = torch.tensor(2 * torch.rand((2, 2)) - 1, requires_grad=True)  # To input, to skjulte nevroner
        self.b1 = torch.tensor(2 * torch.rand((2,)) - 1, requires_grad=True)    # Bias for skjult lag
        self.W2 = torch.tensor(2 * torch.rand((2, 1)) - 1, requires_grad=True)  # Skjult lag til output
        self.b2 = torch.tensor(2 * torch.rand((1,)) - 1, requires_grad=True)    # Bias for output

    # Fremover passering: skjult lag + outputlag med sigmoid
    def f(self, x):
        h = torch.sigmoid(x @ self.W1 + self.b1)  # Skjult lag med sigmoid-aktivering
        return torch.sigmoid(h @ self.W2 + self.b2)  # Outputlag med sigmoid-aktivering

    # Binary cross-entropy loss-funksjon
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


# Initialiser modell og optimizer
model = XORModel()
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=0.1)

# Tren modellen
epochs = 10000
loss_values = []

for epoch in range(epochs):
    loss = model.loss(inputs, outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_values.append(loss.item())
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Print de endelige parametrene etter trening
print(f'Endelige W1 = {model.W1.tolist()}, b1 = {model.b1.tolist()}, W2 = {model.W2.tolist()}, b2 = {model.b2.tolist()}')

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

# Plott prediksjonsplanet
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Plott de faktiske datapunktene og modellens prediksjoner
with torch.no_grad():
    predicted = model.f(inputs).numpy()

for i in range(4):
    ax.scatter(inputs[i, 0].item(), inputs[i, 1].item(), outputs[i].item(), color='r', s=100, label='True output' if i == 0 else "")
    ax.scatter(inputs[i, 0].item(), inputs[i, 1].item(), predicted[i].item(), color='g', marker='x', s=200, label='Model prediction' if i == 0 else "")

ax.set_xlabel('Input A')
ax.set_ylabel('Input B')
ax.set_zlabel('Output / Prediction')
plt.title('3D Model of XOR Operator with Prediction Plane')
plt.legend()
plt.show()
