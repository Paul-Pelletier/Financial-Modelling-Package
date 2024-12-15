# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = x.pow(3) + 0.3 * torch.rand(x.size())

# Create DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model, optimizer, and loss function
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='./runs/simple_nn')

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
    
    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Log model weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(f'{name}/weights', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}/gradients', param.grad, epoch)

# Log the model graph
dummy_input = torch.zeros((1, 1))
writer.add_graph(model, dummy_input)

# Log a custom plot
fig, ax = plt.subplots()
ax.scatter(x.numpy(), y.numpy(), label="Data", alpha=0.6)
ax.plot(x.numpy(), model(x).detach().numpy(), label="Prediction", color="red")
ax.legend()
writer.add_figure('Prediction vs Data', fig)

# Close the writer
writer.close()

print("Training complete. Launch TensorBoard with: tensorboard --logdir=./runs")
