import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate synthetic SVI smile data for multiple expiries
def generate_synthetic_svi_data(n_expiries, n_strikes):
    expiries = np.linspace(0.1, 2.0, n_expiries)  # Expiries from 0.1 to 2.0 years
    strikes = np.linspace(-1, 1, n_strikes)  # Strikes in log-moneyness
    true_params = []
    vol_data = []
    
    for T in expiries:
        a = 0.05 + 0.02 * np.random.rand() + 0.01 * T
        b = 0.15 + 0.1 * np.random.rand()
        rho = -0.5 + 0.3 * np.random.rand()
        m = -0.2 + 0.4 * np.random.rand()
        sigma = 0.1 + 0.2 * np.random.rand()
        
        true_params.append((a, b, rho, m, sigma))
        vol = a + b * (rho * (strikes - m) + np.sqrt((strikes - m)**2 + sigma**2))
        vol_data.append(vol + 0.01 * np.random.randn(*vol.shape))  # Add noise
    
    return strikes, expiries, np.array(vol_data), np.array(true_params)

# SVI model function
def svi_model(params, strikes):
    a, b, rho, m, sigma = torch.chunk(params, chunks=5, dim=-1)
    return a + b * (rho * (strikes - m) + torch.sqrt((strikes - m)**2 + sigma**2))

# Loss function: Mean Squared Error
def svi_loss(params, strikes, vol_data):
    model_vol = svi_model(params, strikes)
    return torch.mean((model_vol - vol_data) ** 2)

# Fit multiple SVI smiles
def fit_svi_smiles(strikes, expiries, vol_data, n_epochs=200, learning_rate=0.1, tolerance=1e-6, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_expiries = len(expiries)
    params = torch.rand((n_expiries, 5), dtype=torch.float32, device=device).requires_grad_()  # Leaf tensor
    params.data *= 0.3  # Scale initialization without breaking "leaf" status
    
    strikes_tensor = torch.tensor(strikes, dtype=torch.float32, device=device).unsqueeze(0)
    vol_data_tensor = torch.tensor(vol_data, dtype=torch.float32, device=device)
    
    optimizer = optim.Adam([params], lr=learning_rate)
    prev_loss = float('inf')
    no_improve_epochs = 0

    start_time = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.stack([
            svi_loss(params[i], strikes_tensor, vol_data_tensor[i]) for i in range(n_expiries)
        ]))
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        
        # Check for early stopping
        if abs(prev_loss - loss_value) < tolerance:
            no_improve_epochs += 1
        else:
            no_improve_epochs = 0
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch} with loss {loss_value:.6f}")
            break
        
        prev_loss = loss_value

        # Logging every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss_value:.6f}")

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return params.cpu().detach()

# Main workflow
n_expiries = 4
n_strikes = 200
strikes, expiries, vol_data, true_params = generate_synthetic_svi_data(n_expiries, n_strikes)

start = time.time()
# Fit SVI smiles
print("Fitting SVI smiles...")
fitted_params = fit_svi_smiles(strikes, expiries, vol_data)

# Plot the results
print("Plotting the results...")
fitted_params_np = fitted_params.numpy()
print("Elapsed time: ", time.time() - start)

# Plot results
n_rows = 4
n_cols = int(np.ceil(n_expiries / n_rows))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), sharey=True)
axes = axes.flatten()

for i, T in enumerate(expiries):
    ax = axes[i]
    strikes_tensor = torch.tensor(strikes, dtype=torch.float32)
    fitted_params_tensor = torch.tensor(fitted_params_np[i], dtype=torch.float32)  # Convert to tensor
    
    ax.scatter(strikes, vol_data[i], label="Noisy Data", color='blue', alpha=0.6)
    ax.plot(
        strikes,
        svi_model(fitted_params_tensor, strikes_tensor).numpy(),  # Use tensor for svi_model
        label="Fitted SVI",
        color='red'
    )
    ax.set_title(f"Expiry {T:.2f} Years")
    ax.set_xlabel("Log-Moneyness")
    if i % n_cols == 0:
        ax.set_ylabel("Implied Volatility")
    ax.legend()
    ax.grid()


for j in range(len(expiries), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
