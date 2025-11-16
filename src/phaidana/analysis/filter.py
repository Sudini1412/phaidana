import numpy as np
from scipy.ndimage.filters import uniform_filter1d


class Filter:
    def __init__(self):
        print('Reconstruction Algorithms: Activated') 
    
    # compute the rolling median over wfs of an event
    def running_mean(self, wfs, gate=100):
        if wfs.ndim > 1: return uniform_filter1d(wfs, size=gate, axis=1)
        return  uniform_filter1d(wfs, size=gate)
    
    # substract mean baseline inside the daq
    def get_baseline(self, wfs, gate=500, start=0):
        if wfs.ndim > 1: return np.mean(wfs[:,start:start+gate], axis=1).reshape((-1, 1))
        return np.mean(wfs[start:start+gate])

