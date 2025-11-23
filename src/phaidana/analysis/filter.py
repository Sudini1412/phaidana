import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import find_peaks

class Filter:
    def __init__(self):
        print('Reconstruction Algorithms: Activated') 
    
    # compute the rolling median over wfs of an event
    def running_mean(self, wfs, gate=20):
        if wfs.ndim > 1: return uniform_filter1d(wfs, size=gate, axis=1)
        return  uniform_filter1d(wfs, size=gate)
    
    # substract mean baseline inside the daq
    def pulse_in_pretrig(self,wfs,gate=250, start=0):
        peaks = []
        wf_cut= wfs[:,start:start+gate]

        for wf in wf_cut:
            p, _ = find_peaks(wf, height=200)
            peaks.append(p)
        peaks = np.array(peaks, dtype=object)
        peaks = peaks.flatten()
        return peaks

    def get_baseline(self, wfs, gate=400, start=0):
        if wfs.ndim > 1: 
            return np.mean(wfs[:,start:start+gate], axis=1).reshape((-1, 1))
        return np.mean(wfs[start:start+gate])

