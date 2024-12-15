import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Generate synthetic SVI smile data for multiple expiries
def generate_synthetic_svi_data(n_expiries, n_strikes):
    expiries = np.linspace(0.1, 2.0, n_expiries)  # Expiries from 0.1 to 2.0 years
    strikes = np.linspace(-1, 1, n_strikes)  # Strikes in log-moneyness
    true_params = []
    vol_data = []
    
    for T in expiries:
        # Increase variability in parameter generation
        a = 0.05 + 0.02 * np.random.rand() + 0.01 * T  # Slight trend with expiry
        b = 0.15 + 0.1 * np.random.rand()             # Wider range for b
        rho = -0.5 + 0.3 * np.random.rand()           # Allow more variation in rho
        m = -0.2 + 0.4 * np.random.rand()             # Center shift for m
        sigma = 0.1 + 0.2 * np.random.rand()          # More diversity in sigma
        
        true_params.append((a, b, rho, m, sigma))
        
        # Generate the smile based on these parameters
        vol = a + b * (rho * (strikes - m) + np.sqrt((strikes - m)**2 + sigma**2))
        vol_data.append(vol + 0.01 * np.random.randn(*vol.shape))  # Add noise
    
    return strikes, expiries, np.array(vol_data), np.array(true_params)

# Loss function: Mean Squared Error between observed and modeled volatilities
@tf.function
def svi_model(params, strikes):
    a, b, rho, m, sigma = tf.split(params, num_or_size_splits=5, axis=-1)
    return a + b * (rho * (strikes - m) + tf.sqrt((strikes - m)**2 + sigma**2))

@tf.function
def svi_loss(params, strikes, vol_data):
    model_vol = svi_model(params, strikes)
    return tf.reduce_mean((model_vol - vol_data) ** 2)

# Fit multiple SVI smiles
def fit_svi_smiles(strikes, expiries, vol_data, n_epochs=200, learning_rate=0.1, tolerance=1e-6, patience=10):
    """
    Fit SVI smiles with early stopping based on loss progression.

    Parameters:
        strikes (array): Array of log-moneyness strikes.
        expiries (array): Array of expiries.
        vol_data (array): Array of observed volatilities.
        n_epochs (int): Maximum number of epochs.
        learning_rate (float): Learning rate for the optimizer.
        tolerance (float): Minimum change in loss to continue training.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        params (tf.Variable): Fitted parameters for all expiries.
    """
    n_expiries = len(expiries)
    params = tf.Variable(tf.random.uniform((n_expiries, 5), 0.01, 0.3), dtype=tf.float32)  # Initial guesses
    
    strikes_tf = tf.constant(strikes, dtype=tf.float32)
    vol_data_tf = tf.constant(vol_data, dtype=tf.float32)
    
    optimizer = tf.optimizers.Adam(learning_rate)
    prev_loss = float('inf')  # Initialize previous loss to infinity
    no_improve_epochs = 0     # Count epochs with no significant improvement
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean([
                svi_loss(params[i], strikes_tf, vol_data_tf[i]) for i in range(n_expiries)
            ])
        gradients = tape.gradient(loss, [params])
        optimizer.apply_gradients(zip(gradients, [params]))
        return loss
    
    for epoch in range(n_epochs):
        loss = train_step().numpy()
        
        # Check for early stopping
        if abs(prev_loss - loss) < tolerance:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0  # Reset if there's improvement
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch} with loss {loss:.6f}")
            break
        
        prev_loss = loss
        
        # Logging every 50 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    return params


# Main workflow
n_expiries = 4
n_strikes = 200
strikes, expiries, vol_data, true_params = generate_synthetic_svi_data(n_expiries, n_strikes)

# Convert vol_data to float32 for TensorFlow compatibility
vol_data = vol_data.astype(np.float32)

start = time.time()
# Fit SVI smiles
print("Fitting SVI smiles...")
fitted_params = fit_svi_smiles(strikes, expiries, vol_data)

# Plot the results with 4 rows
print("Plotting the results...")
fitted_params_np = fitted_params.numpy()
print("Elapsed time "+ str(time.time()-start))
# Calculate number of columns based on expiries and rows
n_rows = 4
n_cols = int(np.ceil(n_expiries / n_rows))  # Compute columns needed for 4 rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), sharey=True)
axes = axes.flatten()  # Flatten the axes array for easy indexing

for i, T in enumerate(expiries):
    ax = axes[i]
    # Convert strikes to TensorFlow tensor for compatibility
    strikes_tf = tf.constant(strikes, dtype=tf.float32)
    ax.scatter(strikes, vol_data[i], label="Noisy Data", color='blue', alpha=0.6)
    ax.plot(
        strikes, 
        svi_model(fitted_params_np[i], strikes_tf).numpy(), 
        label="Fitted SVI", 
        color='red'
    )
    ax.set_title(f"Expiry {T:.2f} Years")
    ax.set_xlabel("Log-Moneyness")
    if i % n_cols == 0:
        ax.set_ylabel("Implied Volatility")
    ax.legend()
    ax.grid()

# Hide empty subplots if any
for j in range(len(expiries), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

