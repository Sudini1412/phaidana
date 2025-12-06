import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

class PulseFitter:
    def expo(self, x, a, b):
        # x: time (microseconds)
        # a: Amplitude (Volts)
        # b: Decay constant (microseconds)
        return a * np.exp(-(x / b))

    def fit_expo(self, x, y, p0=None, y_errs=None, abs_errs=True):
        # 1. FIX: Ensure p0 matches the 2 parameters (a, b) of expo()
        if p0 is None:
            # Guess 'a': The max voltage in the array
            guess_a = np.max(y) 
            # Guess 'b': A rough estimate. 
            # If your pulses are usually 10-100us long, 20 is a safe start.
            guess_b = 20.0  
            p0 = (guess_a, guess_b)
        
        try:
            coeffs, covariances = scipy.optimize.curve_fit(
                self.expo, x, y, p0=p0, sigma=y_errs, absolute_sigma=abs_errs
            )
            # Returns [a, b] and their error margins
            return coeffs, np.sqrt(np.diag(covariances))
        except RuntimeError:
            print("Fit failed to converge.")
            return None, None

# --- Implementation ---

# 1. Create Dummy Data (Simulating a Pulse)
# Time in microseconds (0 to 100)
time_us = np.linspace(0, 100, 500) 
# True parameters: 5 Volts amplitude, 15us decay
true_a = 5.0
true_b = 15.0 
# Generate voltage with some noise
volts = true_a * np.exp(-(time_us / true_b)) + np.random.normal(0, 0.1, len(time_us))

# 2. Initialize your class
fitter = PulseFitter()

# 3. Perform the fit
# Note: If your pulse has a rising edge, SLICE the data to start at the peak!
# Example: peak_index = np.argmax(volts)
# fit_params, fit_errs = fitter.fit_expo(time_us[peak_index:], volts[peak_index:])

params, errors = fitter.fit_expo(time_us, volts)

# 4. Results
if params is not None:
    a_fit, b_fit = params
    print(f"Fitted Amplitude (a): {a_fit:.3f} V")
    print(f"Fitted Decay Constant (b): {b_fit:.3f} µs")

    # 5. Visualization
    fitted_curve = fitter.expo(time_us, *params)
    plt.scatter(time_us, volts, s=5, label='Raw Data', color='gray')
    plt.plot(time_us, fitted_curve, 'r-', linewidth=2, label='Fit')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude (V)')
    plt.legend()
    plt.show()