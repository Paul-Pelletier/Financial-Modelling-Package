import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())
#tf.debugging.set_log_device_placement(True)  # Logs operations on devices
import sys

# Get the Python version
python_version = sys.version

# Print the version
print(f"Python version: {python_version}")

# Alternatively, you can get a tuple of the major, minor, and micro versions
python_version_tuple = sys.version_info
print(f"Python version tuple: {python_version_tuple.major}.{python_version_tuple.minor}.{python_version_tuple.micro}")


# Generate synthetic dataset
x_train = tf.random.normal([100000, 1])  # Simplified to 1 feature for easy visualization
y_train = 10 * x_train**3 + 3 * x_train**2 + tf.random.normal([100000, 1])  # y = x^3 + 3x^2 + noise

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(1,)),  # Input shape for 1 feature
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Create a tf.data.Dataset
batch_size = 200
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Train the model using a custom training loop
epochs = 5
print("Starting training...")
start_time = time.time()

# Lists to store epoch-wise loss for plotting
epoch_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    batch_count = 0

    for batch, (x_batch, y_batch) in enumerate(dataset):
        batch_count += 1
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.MSE(y_batch, predictions)
            batch_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(batch_loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Accumulate epoch loss and print batch progress
        epoch_loss += batch_loss.numpy()
        print(f"Batch {batch + 1}/{len(dataset)} - Loss: {batch_loss.numpy():.4f}")

    # Average epoch loss
    epoch_loss /= batch_count
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds.")

# Make predictions
y_pred = model.predict(x_train)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Training Data and Model Predictions
plt.subplot(2, 1, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data", alpha=0.6)
plt.scatter(x_train.numpy(), y_pred, label="Model Predictions", color="red", alpha=0.6)
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
