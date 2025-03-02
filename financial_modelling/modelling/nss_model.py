import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class NelsonSiegelSvensson:
    """
    Nelson-Siegel-Svensson (NSS) yield curve model.
    
    Parameters:
    - beta0: Long-term level of the forward rate
    - beta1: Short-term component
    - beta2: Medium-term hump
    - beta3: Long-term hump
    - lambda1: Decay factor for the first hump
    - lambda2: Decay factor for the second hump
    """
    
    def __init__(self, beta0=0.02, beta1=-0.03, beta2=0.04, beta3=-0.02, lambda1=5, lambda2=15):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    @staticmethod
    def forward_curve(t, beta0, beta1, beta2, beta3, lambda1, lambda2):
        """
        Nelson-Siegel-Svensson forward rate function.
        """
        term1 = beta0
        term2 = beta1 * np.exp(-t / lambda1)
        term3 = beta2 * (t / lambda1) * np.exp(-t / lambda1)
        term4 = beta3 * (t / lambda2) * np.exp(-t / lambda2)
        return term1 + term2 + term3 + term4

    def fit(self, maturities, forward_rates):
        """
        Fit the NSS model to observed forward rates.
        """
        # Initial guess
        initial_params = [self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2]

        # Fit using scipy.optimize.curve_fit
        params, _ = opt.curve_fit(self.forward_curve, maturities, forward_rates, p0=initial_params, maxfev=10_000)

        # Update parameters
        self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2 = params
        return params

    def predict(self, maturities):
        """
        Predict forward rates for given maturities.
        """
        return self.forward_curve(maturities, self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2)

    def plot(self, maturities, observed_rates=None):
        """
        Plot the NSS forward curve.
        """
        predicted_rates = self.predict(maturities)

        plt.figure(figsize=(10, 5))
        plt.plot(maturities, predicted_rates, label="Fitted NSS Curve", linewidth=2)
        if observed_rates is not None:
            plt.scatter(maturities, observed_rates, color='red', label="Observed Data")
        #plt.axvline(x=self.lambda1, color='red', linestyle="--", label=f"1st Hump (λ1={self.lambda1:.2f})")
        #plt.axvline(x=self.lambda2, color='blue', linestyle="--", label=f"2nd Hump (λ2={self.lambda2:.2f})")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Forward Rate")
        plt.title("Nelson-Siegel-Svensson Forward Curve")
        plt.legend()
        plt.grid()
        plt.show()
