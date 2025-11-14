import numpy as np 
import scipy.optimize
from scipy.stats import moyal

class Fit:

    def __init__(self):
        return

    def expo(self, x, a, b):
        
        return (a * np.exp(-(x / b)))


    def fit_expo(self, x, y, p0 = None, y_errs = None, abs_errs = True):
        
        # This fit should always have some kind of default values, so if none are specified, generate them
        if p0 is None:
            p0 = (np.min(y), 16.0, np.min(y) / 100.0, 1500)
        
        try:
            coeffs, covariances = scipy.optimize.curve_fit(self.expo, x, y, p0 = p0, sigma = y_errs, absolute_sigma = abs_errs)
            
            return coeffs, np.sqrt(np.diag(covariances))
        
        except RuntimeError:
            print(f" fit_function_SciPy.fit_expo() - cannot fit exponential function to this event!")
            
            return None

    def moyal_fit(self, x, loc, scale, A):
        return A * moyal.pdf(x, loc=loc, scale=scale)



    def fit_moyal(self, x, y, p0=None, y_errs=None, abs_errs=True):
        if p0 is None:
            loc_guess = x[np.argmax(y)]
            scale_guess = (max(x) - min(x)) / 10
            A_guess = np.max(y)
            p0 = (loc_guess, scale_guess, A_guess)
        try:
            coeffs, cov = scipy.optimize.curve_fit(
                self.moyal_fit,
                x,
                y,
                p0=p0,
                sigma=y_errs,
                absolute_sigma=abs_errs if y_errs is not None else False,
            )
        
            errors = np.sqrt(np.diag(cov))

            return coeffs, errors
            
        except RuntimeError:
            print("Error fitting Landau")
            return None