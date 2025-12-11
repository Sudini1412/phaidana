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
    
    def get_baseline(self, wfs, gate=250, start=0):
        """
        Calculates baseline using MEDIAN. 
        Essential if the baseline region might contain pulses.
        """
        # Slice the region
        region = wfs[:, start:start+gate]
        
        # Calculate Median along the time axis (axis 1)
        # We reshape to (N, 1) to allow easy broadcasting later
        baseline = np.median(region, axis=1)[:, np.newaxis]
        return baseline
    
    def get_pretrig_mask(self, wfs, threshold, gate=250, start=0):
        """
        Returns a boolean mask.
        Rejects events based on PRE-TRIGGER stability.
        
        Args:
            wfs: Waveforms (2D array)
            threshold: 
                If using 'std': Max allowed standard deviation (e.g., 5-10 ADC counts)
                If using 'abs': Max allowed amplitude deviation
        """
        # 1. Slice the pre-trigger
        region = wfs[:, start:start+gate]
        
        # 2. Check Standard Deviation (Flatness)
        # Your clean baseline might have std ~2-3. This tail event has std ~1000.
        # This catches slopes AND peaks.
        stds = np.std(region, axis=1)
        
        # 3. Create Mask
        # If std is too high, it's not a flat baseline -> False (Dirty)
        # Note: You might need to tune 'threshold' to be slightly above your noise floor.
        mask = stds < threshold
        
        return mask
    
    # substract mean baseline inside the daq
    def pulse_in_pretrig(self,wfs, threshold, gate=250, start=0):
        peaks = []
        wf_cut= wfs[:,start:start+gate]

        for wf in wf_cut:
            p, _ = find_peaks(wf, height=threshold)
            peaks.append(p)
        peaks = np.array(peaks, dtype=object)
        peaks = peaks.flatten()
        return peaks

    def get_baseline(self, wfs, gate=400, start=0):
        if wfs.ndim > 1: 
            return np.mean(wfs[:,start:start+gate], axis=1).reshape((-1, 1))
        return np.mean(wfs[start:start+gate])

