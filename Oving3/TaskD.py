import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Last inn Fashion MNIST dataset
fashion_mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                                        transform=transforms.ToTensor())
x_train = fashion_mnist_train.data.reshape(-1, 1, 28, 28).float()
y_train = torch.zeros((fashion_mnist_train.targets.shape[0], 10))  # Output tensor
y_train[torch.arange(fashion_mnist_train.targets.shape[0]), fashion_mnist_train.targets] = 1  # Populer output

fashion_mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                                       transform=transforms.ToTensor())
x_test = fashion_mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((fashion_mnist_test.targets.shape[0], 10))  # Output tensor
y_test[torch.arange(fashion_mnist_test.targets.shape[0]), fashion_mnist_test.targets] = 1  # Populer output

# Normalisering av inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Del opp treningsdata i batcher
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


# Modellarkitektur
class FashionCNNModel(nn.Module):
    def __init__(self):
        super(FashionCNNModel, self).__init__()

        # Konvolusjons- og poolinglag
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fullt tilkoblet lag
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10) # 10 output noder for hver kategorii av klær

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def logits(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1) # Softmaxfor sannsynligheter for hver klasse

    # Cross entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1)) # Kryssentropi for klassifisering av klær

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float()) # Nøyaktighet

# Opprett modell
model = FashionCNNModel()

# Optimalisering med Adam
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Tren modellen
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Beregn tap
        optimizer.step()  # Utfør optimalisering
        optimizer.zero_grad()  # Nullstill gradienter

    print("accuracy = %s" % model.accuracy(x_test, y_test))

# accuracy = tensor(0.9193)