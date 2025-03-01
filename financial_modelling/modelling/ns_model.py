import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class NelsonSiegel:
    """
    Nelson-Siegel- (NSS) yield curve model.
    
    Parameters:
    - beta0: Long-term level of the forward rate
    - beta1: Short-term component
    - beta2: Long-term hump
    - lambda1: Decay factor for the first hump
    """
    
    def __init__(self, beta0=2000, beta1=2000, beta2=2000, lambda1=0.1):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda1 = lambda1
        self.observed_rates = None
        self.observed_maturities = None

    @staticmethod
    def forward_curve(t, beta0, beta1, beta2, lambda1):
        """
        Nelson-Siegel- forward rate function.
        """
        term1 = beta0
        term2 = beta1 * np.exp(t / lambda1)
        term3 = beta2 * (t / lambda1) * np.exp(t / lambda1)
        return term1 + term2 + term3

    def fit(self, maturities, forward_rates):
        """
        Fit the NSS model to observed forward rates.
        """
        # Initial guess
        initial_params = [self.beta0, self.beta1, self.beta2, self.lambda1]


        self.observed_maturities = maturities
        self.observed_rates = forward_rates
        # Fit using scipy.optimize.curve_fit
        params, _ = opt.curve_fit(self.forward_curve, maturities, forward_rates, p0=initial_params, maxfev=10000)

        # Update parameters
        self.beta0, self.beta1, self.beta2, self.lambda1 = params
        return params

    def predict(self, maturities):
        """
        Predict forward rates for given maturities.
        """
        return self.forward_curve(maturities, self.beta0, self.beta1, self.beta2, self.lambda1)

    def plot(self, maturities, observed_rates=None):
        """
        Plot the NSS forward curve.
        """
        predicted_rates = self.predict(maturities)

        plt.figure(figsize=(10, 5))
        plt.plot(maturities, predicted_rates, label="Fitted NS Curve", linewidth=2)
        if observed_rates is not None:
            plt.scatter(self.observed_maturities, self.observed_rates, color='red', label="Observed Data")
        #plt.axvline(x=self.lambda1, color='red', linestyle="--", label=f"1st Hump (λ1={self.lambda1:.2f})")
        #plt.axvline(x=self.lambda2, color='blue', linestyle="--", label=f"2nd Hump (λ2={self.lambda2:.2f})")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Term Structure")
        plt.title("Nelson-Siegel parameterization")
        plt.legend()
        plt.grid()
        plt.show()
