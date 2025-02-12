import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ExtendedConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ExtendedConvolutionalNeuralNetworkModel, self).__init__()

        # Første konvolusjonslag
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Andre konvolusjonslag nr 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # dense lag
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Legger til et nytt dense-lag
        self.fc2 = nn.Linear(128, 10)  # Output lag med 10 noder

    def logits(self, x):
        # Passerer gjennom første konvolusjonslag og pooling
        x = self.pool(self.conv1(x)) # Reduserer fra 28x28 til 14x14
        # Passerer gjennom andre konvolusjonslag og pooling
        x = self.pool2(self.conv2(x)) # Reduserer fra 14x14 til 7x7
        # Flatt ut tensoren før den går inn i det fullt tilkoblede laget
        x = x.view(-1, 64 * 7 * 7)
        # Fullt tilkoblet lag
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1) # Softmax på output for sannsynligheter

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1)) # Kryssentropi for klassifisering av sifre

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = ExtendedConvolutionalNeuralNetworkModel()

# Optimize
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))

 # v1 = accuracy = tensor(0.9782)
# v2 = accuracy = tensor(0.9843)

