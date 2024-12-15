import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch is using CUDA: {torch.cuda.is_available()}")

# Generate synthetic dataset
x_train = torch.randn(100000, 1, device=device)  # 1 feature
y_train = 10 * x_train**3 + 3 * x_train**2 + torch.randn(100000, 1, device=device)  # y = x^3 + 3x^2 + noise

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel().to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Create DataLoader for batching
batch_size = 200
dataset = torch.utils.data.TensorDataset(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
epochs = 5
print("Starting training...")
start_time = time.time()

epoch_losses = []  # Store epoch-wise loss for plotting

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    batch_count = 0

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        batch_count += 1
        optimizer.zero_grad()
        predictions = model(x_batch)
        batch_loss = loss_fn(predictions, y_batch)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {batch_loss.item():.4f}")

    epoch_loss /= batch_count
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds.")

# Make predictions
y_pred = model(x_train).detach().cpu().numpy()

# Plotting
x_train_np = x_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()

plt.figure(figsize=(12, 10))

# Plot 1: Training Data and Model Predictions
plt.subplot(2, 1, 1)
plt.scatter(x_train_np, y_train_np, label="Training Data", alpha=0.6)
plt.scatter(x_train_np, y_pred, label="Model Predictions", color="red", alpha=0.6)
plt.title("Training Data and Model Predictions")
plt.xlabel("Input Feature (x_train)")
plt.ylabel("Target Value (y_train / y_pred)")
plt.legend()
plt.grid()

# Plot 2: Loss Function Across Epochs
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), epoch_losses, label="Training Loss", marker='o')
plt.title("Loss Function Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()
