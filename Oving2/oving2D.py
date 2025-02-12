import torch
import torchvision
import matplotlib.pyplot as plt
import os

# Last ned og klargjør MNIST-datasettet
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)

# Flat ut bilder til 1D-vektor med 784 elementer (28x28), og normaliser til float
x_train = mnist_train.data.reshape(-1, 784).float() / 255.0  # Normalisering
x_test = mnist_test.data.reshape(-1, 784).float() / 255.0    # Normalisering

# Lag en modell med softmax
class SoftmaxModel(torch.nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.linear = torch.nn.Linear(784, 10)  # 784 input features (28x28), 10 output classes

    def forward(self, x):
        return torch.nn.functional.softmax(self.linear(x), dim=1)  # Softmax for output

# Initialiser modell, tapfunksjon og optimizer
model = SoftmaxModel()
criterion = torch.nn.CrossEntropyLoss()  # Krysstap er egnet for klassifisering
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # Prøv en høyere læringsrate for raskere konvergens

# Tren modellen
epochs = 1000  # Begrens til 1000 epoker (vi stopper tidlig ved 90% nøyaktighet)
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, mnist_train.targets)
    loss.backward()
    optimizer.step()

    # Evaluer nøyaktigheten på treningssettet
    with torch.no_grad():
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == mnist_train.targets).sum().item()
        accuracy = correct / mnist_train.targets.size(0)

    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}')

    # Stopp tidlig hvis nøyaktigheten er over 0.9
    if accuracy >= 0.9:
        print(f'Nøyaktighet på {accuracy} oppnådd. Treningen stoppes.')
        break

# Evaluer på testsettet
with torch.no_grad():
    y_pred_test = model(x_test)
    _, predicted_test = torch.max(y_pred_test, 1)
    correct_test = (predicted_test == mnist_test.targets).sum().item()
    test_accuracy = correct_test / mnist_test.targets.size(0)

print(f'Nøyaktighet på testsettet: {test_accuracy}')

# Lagre vektene som bilder etter optimalisering
os.makedirs('weights', exist_ok=True)
with torch.no_grad():
    weights = model.linear.weight.detach().numpy()

# Vektene har formen (784, 10), så vi kan visualisere hver kolonne (tilhørende hver klasse)
for i in range(10):
    plt.imshow(weights[i].reshape(28, 28), cmap='gray')
    plt.title(f'Vekter for siffer {i}')
    plt.savefig(f'weights/weight_{i}.png')
    plt.close()
