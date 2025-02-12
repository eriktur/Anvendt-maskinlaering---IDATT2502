import pandas as pd
import torch
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/erik/Desktop/NTNU/AnvendtMaskinlæring/Oving1/day_head_circumference.csv')


if 'day' in data.columns and 'head circumference' in data.columns:
    age = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)
    head_circumference = torch.tensor(data['head circumference'].values, dtype=torch.float32).reshape(-1, 1)
else:
    print("Column names 'day' and 'head circumference' not found in the CSV file.")
    exit()

class NonLinearRegressionModel:
    def __init__(self):

        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)


    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31


    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = NonLinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], lr=0.000001)


for epoch in range(220000):
    loss = model.loss(age, head_circumference)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


print(f"Final W = {model.W.item()}, b = {model.b.item()}, loss = {loss.item()}")


plt.figure('Nonlinear regression 2D')
plt.title('Predict head circumference based on age')
plt.xlabel('Age (days)')
plt.ylabel('Head Circumference (cm)')


plt.scatter(age.numpy(), head_circumference.numpy(), label='Observed data')


x_range = torch.linspace(torch.min(age), torch.max(age), steps=100).reshape(-1, 1)
predicted_circumference = model.f(x_range).detach()
plt.plot(x_range.numpy(), predicted_circumference.numpy(), color='orange',
         label='Predicted Head Circumference')

plt.legend()
plt.show()

# Final W = 0.0027864747680723667, b = -0.18753454089164734, loss = 2.6035995483398438