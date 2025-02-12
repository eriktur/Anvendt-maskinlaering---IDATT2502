import pandas as pd
import torch
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/erik/Desktop/NTNU/AnvendtMaskinlæring/Oving1/length_weight.csv')


if 'length' in data.columns and 'weight' in data.columns:
    length = torch.tensor(data['length'].values, dtype=torch.float32).reshape(-1, 1)
    weight = torch.tensor(data['weight'].values, dtype=torch.float32).reshape(-1, 1)
else:
    print("Column names 'length' and 'weight' not found in the CSV file.")


print(torch.isnan(length).any(), torch.isinf(length).any())
print(torch.isnan(weight).any(), torch.isinf(weight).any())

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Loss function: Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()


optimizer = torch.optim.SGD([model.W, model.b], lr=0.001)
for epoch in range(1000000):
    model.loss(length, weight).backward()
    torch.nn.utils.clip_grad_norm_([model.W, model.b], max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10000 == 0:
        print(f'Epoch {epoch}, Loss: {model.loss(length, weight).item()}')

# Print model variables and loss
print(f"W = {model.W.item():.4f}, b = {model.b.item():.4f}, loss = {model.loss(length, weight).item():.4f}")



plt.plot(length.numpy(), weight.numpy(), 'o', label='Observed data')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (kg)')
x = torch.tensor([[torch.min(length)], [torch.max(length)]])  # Min and max length for the line plot
plt.plot(x.numpy(), model.f(x).detach().numpy(), label='f(x) = xW+b')
plt.legend()
plt.show()

# W = 0.2380, b = -8.5242, loss = 0.9960