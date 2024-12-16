import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

# Initialize TensorBoard writer
log_dir = "runs/simple_nn"
writer = SummaryWriter(log_dir=log_dir)

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Generate synthetic data
x = torch.randn(1000, 1)
y = 10 * x**3 + 3 * x**2 + 2 + torch.randn(1000, 1) * 0.5

# Training setup
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with TensorBoard logging
epochs = 100
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Log training loss to TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Log final model graph
writer.add_graph(model, x)

# Close the writer
writer.close()
