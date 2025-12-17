import numpy as np 
import scipy.optimize
from scipy.stats import moyal
import matplotlib.pyplot as plt 

class Fit:

    def __init__(self):
        return

    def expo(self, x, a, b):
        # x: time (microseconds)
        # a: Amplitude (Volts)
        # b: Decay constant (microseconds)
        return (a * np.exp(-(x / b)))


    def fit_expo(self, x, y, p0 = None, y_errs = None, abs_errs = True):
        
        # This fit should always have some kind of default values, so if none are specified, generate them
        if p0 is None:
            guess_a = np.max(y)
            guess_b = 1.6
            p0 = (guess_a, guess_b)
        
        try:
            coeffs, covariances = scipy.optimize.curve_fit(self.expo, x, y, p0 = p0, sigma = y_errs, absolute_sigma = abs_errs)
            
            return coeffs, np.sqrt(np.diag(covariances))
        
        except RuntimeError:
            return None, None
        