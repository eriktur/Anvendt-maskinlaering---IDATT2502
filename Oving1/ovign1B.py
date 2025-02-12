import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv('/Users/erik/Desktop/NTNU/AnvendtMaskinlæring/Oving1/day_length_weight.csv')

# Ensure the column names are correct
if 'day' in data.columns and 'length' in data.columns and 'weight' in data.columns:
    length = torch.tensor(data['length'].values, dtype=torch.float32).reshape(-1, 1)
    weight = torch.tensor(data['weight'].values, dtype=torch.float32).reshape(-1, 1)
    age = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)

    # Combine length and weight into a single input tensor
    inputs = torch.cat((length, weight), dim=1)

    # Standardize the inputs (length and weight)
    scaler_x = StandardScaler()
    inputs_normalized = torch.tensor(scaler_x.fit_transform(inputs), dtype=torch.float32)

    # Standardize the output (age)
    scaler_y = StandardScaler()
    age_normalized = torch.tensor(scaler_y.fit_transform(age), dtype=torch.float32)
else:
    print("Column names 'day', 'length', and 'weight' not found in the CSV file.")
    exit()

class LinearRegressionModel3D:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # Two weights for length and weight
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Loss function: Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel3D()
optimizer = torch.optim.SGD([model.W, model.b], lr=0.0001)

# Training loop
for epoch in range(100000):
    loss = model.loss(inputs_normalized, age_normalized)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print loss periodically to monitor progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Print final model variables and loss
print(f"Final W = {model.W}, b = {model.b}, loss = {loss.item()}")

# Visualize the result
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

# Plot the observed data points with the original values
ax.scatter(inputs[:, 0].numpy(), inputs[:, 1].numpy(), age.numpy(), color='blue', s=50, label='Observed data')

# Create a grid of values for length and weight to predict the age using the original scale
length_range = torch.linspace(torch.min(inputs[:, 0]), torch.max(inputs[:, 0]), steps=100)
weight_range = torch.linspace(torch.min(inputs[:, 1]), torch.max(inputs[:, 1]), steps=100)
length_grid, weight_grid = torch.meshgrid(length_range, weight_range, indexing='ij')

# Combine length and weight grids into a single input tensor and normalize
grid_inputs = torch.cat((length_grid.reshape(-1, 1), weight_grid.reshape(-1, 1)), dim=1)
grid_inputs_normalized = torch.tensor(scaler_x.transform(grid_inputs), dtype=torch.float32)

# Predict the age for the grid
age_pred_normalized = model.f(grid_inputs_normalized).reshape(length_grid.shape)

# Rescale the predicted age back to the original scale
age_pred = scaler_y.inverse_transform(age_pred_normalized.detach().numpy())

# Plot the predicted surface
ax.plot_wireframe(length_grid.numpy(), weight_grid.numpy(), age_pred, color='orange', alpha=0.5, label='Predicted surface')

ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Age (days)')
ax.view_init(elev=20, azim=120)  # Set a good viewing angle
plt.legend()
plt.show()
