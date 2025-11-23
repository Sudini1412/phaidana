import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto

# ---------------------------------------------------------------------------
# 1. Configuration & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PulseResult:
    """Immutable output structure for a detected pulse."""
    t_start: int
    t_end: int
    peak_index: int
    peak_amplitude: float
    integral: float

@dataclass
class PulseConfig:
    """
    Tuning parameters. 
    Defaults match the original script's logic.
    """
    high_threshold: float = 5.0
    low_threshold: float = 1.0
    avg_gate: int = 5
    min_gap_samples: int = 2
    min_after_peak: int = 3
    valley_depth_threshold: float = 2.0
    valley_rel_fraction: float = 0.3

class State(Enum):
    STANDBY = auto()
    RISING = auto()
    FALLING = auto()

# ---------------------------------------------------------------------------
# 2. Internal State Helper
# ---------------------------------------------------------------------------

class PulseCandidate:
    """
    Tracks the statistics of a specific pulse while it is being analyzed.
    """
    def __init__(self, start_idx: int, start_val: float):
        self.start_idx = start_idx
        self.peak_idx = start_idx
        self.peak_val = start_val
        self.valley_idx = start_idx
        self.valley_val = start_val
    
    def update_peak(self, idx: int, val: float):
        """Called during RISING state."""
        if val > self.peak_val:
            self.peak_val = val
            self.peak_idx = idx
            # Reset valley tracking when a new high is found
            self.valley_val = val
            self.valley_idx = idx

    def update_valley(self, idx: int, val: float):
        """Called during FALLING state."""
        if val < self.valley_val:
            self.valley_val = val
            self.valley_idx = idx

    def finalize(self, end_idx: int, raw_waveform: np.ndarray) -> PulseResult:
        """Freezes the candidate into a final result object."""
        # Ensure indices are within bounds
        safe_end = min(end_idx, len(raw_waveform) - 1)
        safe_start = max(0, self.start_idx)
        
        segment = raw_waveform[safe_start : safe_end + 1]
        
        # If segment is empty (edge case), handle gracefully
        if segment.size == 0:
            peak_amp = 0.0
            integral = 0.0
        else:
            peak_amp = np.max(segment)
            integral = np.sum(segment)

        return PulseResult(
            t_start=safe_start,
            t_end=safe_end,
            peak_index=self.peak_idx,
            peak_amplitude=peak_amp,
            integral=integral
        )

# ---------------------------------------------------------------------------
# 3. Main Processor
# ---------------------------------------------------------------------------

class PulseFinder:
    def __init__(self, config: PulseConfig = PulseConfig()):
        self.cfg = config

    def _smooth_waveform(self, wf: np.ndarray) -> np.ndarray:
        """
        Applies a Semi-Causal Moving Average using Convolution.
        
        - Looks mostly backward (history)
        - Looks 1 sample forward (to catch edges)
        - Maintains the original slight delay (phase shift) so pulses don't 'shift up/left'
        """
        gate = self.cfg.avg_gate
        if gate < 1: return wf
        
        # 1. Create the Moving Average Kernel
        kernel = np.ones(gate) / gate
        
        # 2. Determine Padding to match original 'semi-causal' timing
        # Original logic: Window = [t-(gate-2) ... t ... t+1]
        # For Gate 5: We want the window to be indices [i-3, i-2, i-1, i, i+1]
        # This requires padding: 3 on the Left, 1 on the Right.
        look_ahead = 1
        look_back = gate - 1 - look_ahead
        
        # Ensure we don't have negative padding if gate is tiny
        look_back = max(0, look_back)
        
        padded = np.pad(wf, (look_back, look_ahead), mode='edge')
        
        # 3. Convolve
        # 'valid' convolution with this specific asymmetric padding 
        # produces an output exactly N samples long with the correct phase lag.
        return np.convolve(padded, kernel, mode='valid')

    def _find_pulse_start(self, wf: np.ndarray, current_idx: int, last_pulse_end: int) -> int:
        """
        Rewinds from the trigger point to find the crossing of the low_threshold.
        """
        j = current_idx
        # Rewind while above low threshold
        while j > 0 and wf[j - 1] >= self.cfg.low_threshold:
            j -= 1
        
        # Enforce minimum gap from previous pulse
        min_start = last_pulse_end + self.cfg.min_gap_samples
        return max(j, min_start)

    def _refine_split_point(self, filt_arr: np.ndarray, peak_idx: int, current_idx: int, default_valley: int) -> int:
        """
        Scans the filtered data between the peak and current point to find
        the best 'local minimum' to cut the pulse.
        """
        raw_search_start = peak_idx + 1
        raw_search_end = current_idx
        
        segment = filt_arr[raw_search_start:raw_search_end]
        
        if segment.size < 3:
            return default_valley

        # Identify indices where value is strictly smaller than neighbors
        local_mins = np.where((segment[1:-1] < segment[:-2]) & (segment[1:-1] < segment[2:]))[0]
        
        if local_mins.size > 0:
            # +1 for slicing offset, +1 for the index inside segment
            return raw_search_start + local_mins[-1] + 1
        
        return default_valley

    def process(self, waveform: np.ndarray) -> Tuple[List[PulseResult], np.ndarray]:
        """
        Main entry point. Returns found pulses and the filtered trace used for logic.
        """
        wf_raw = np.asarray(waveform, dtype=float)
        wf_filt = self._smooth_waveform(wf_raw)

        n = len(wf_raw)
        
        pulses: List[PulseResult] = []
        candidate: Optional[PulseCandidate] = None
        state = State.STANDBY
        
        i = 0
        while i < n:
            val = wf_filt[i]

            if state == State.STANDBY:
                if val > self.cfg.high_threshold:
                    # 1. Check previous pulse constraint
                    last_end = pulses[-1].t_end if pulses else -999
                    
                    # 2. Find start time
                    t0 = self._find_pulse_start(wf_raw, i, last_end)
                    
                    if t0 < n:
                        candidate = PulseCandidate(t0, val)
                        state = State.RISING
                    else:
                        break # Waveform ended

            elif state == State.RISING:
                if val > candidate.peak_val:
                    candidate.update_peak(i, val)
                else:
                    state = State.FALLING
                    candidate.update_valley(i, val)

            elif state == State.FALLING:
                candidate.update_valley(i, val)
                
                # --- Check Pile-up (Split Logic) ---
                depth = candidate.peak_val - candidate.valley_val
                rise = val - candidate.valley_val
                
                is_deep = depth > self.cfg.valley_depth_threshold
                is_rising = rise >= (self.cfg.valley_rel_fraction * depth)
                valid_valley = (candidate.valley_idx - candidate.start_idx) > 1
                
                if valid_valley and is_deep and is_rising:
                    # Find the precise cut point
                    split_idx = self._refine_split_point(wf_filt, candidate.peak_idx, i, candidate.valley_idx)
                    
                    # A. Close the FIRST pulse
                    pulses.append(candidate.finalize(split_idx - 1, wf_raw))
                    
                    # B. Start the SECOND pulse (immediately at split_idx)
                    candidate = PulseCandidate(split_idx, val)
                    state = State.RISING
                    
                    # We do NOT increment i here (or we treat it as handled).
                    # The logic processes this sample as the start of the new rise.
                    continue 

                # --- Check Closure (End Logic) ---
                # Must drop below low_threshold AND satisfy minimum width requirement
                if val < self.cfg.low_threshold and i > candidate.peak_idx + self.cfg.min_after_peak:
                    pulses.append(candidate.finalize(i, wf_raw))
                    candidate = None
                    state = State.STANDBY

            i += 1

        # Handle active pulse at end of file
        if candidate is not None:
            pulses.append(candidate.finalize(n - 1, wf_raw))

        return pulses, wf_filt

# ---------------------------------------------------------------------------
# 4. Test & Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate Synthetic Data
    # Create 250 samples
    x = np.linspace(0, 250, 250)
    
    # Add Gaussian Noise
    np.random.seed(42) # Fixed seed for reproducibility
    noise = np.random.normal(0, 0.4, 250)
    
    # Pulse 1: Standard
    p1 = 12 * np.exp(-(x - 40) / 6) * (x > 40)
    
    # Pulse 2 & 3: The Pile-up Scenario
    # Pulse 2 starts at 100, Pulse 3 starts at 115 (on the tail of P2)
    p2 = 10 * np.exp(-(x - 100) / 5) * (x > 100)
    p3 = 8 * np.exp(-(x - 115) / 5) * (x > 115)
    
    raw_data = p1 + p2 + p3 + noise

    # 2. Process
    cfg = PulseConfig(
        high_threshold=3.0,
        low_threshold=1.0,
        valley_rel_fraction=0.2 # Sensitive enough to catch the p2/p3 split
    )
    finder = PulseFinder(cfg)
    found_pulses, filtered_trace = finder.process(raw_data)

    # 3. Console Output
    print(f"Processing complete. Found {len(found_pulses)} pulses.\n")
    print(f"{'#':<3} {'Start':<8} {'End':<8} {'Peak Amp':<10} {'Area':<10}")
    print("-" * 45)
    for i, p in enumerate(found_pulses):
        print(f"{i+1:<3} {p.t_start:<8} {p.t_end:<8} {p.peak_amplitude:<10.2f} {p.integral:<10.2f}")

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Traces
    plt.plot(raw_data, color='gray', alpha=0.5, label='Raw Noisy Signal')
    plt.plot(filtered_trace, color='blue', linewidth=1.5, label='Filtered (Algorithm View)')
    
    # Plot Thresholds
    plt.axhline(cfg.high_threshold, color='red', linestyle='--', alpha=0.5, label='High Thresh')
    plt.axhline(cfg.low_threshold, color='orange', linestyle='--', alpha=0.5, label='Low Thresh')

    # Highlight Pulses
    for i, p in enumerate(found_pulses):
        # Highlight the duration area
        plt.axvspan(p.t_start, p.t_end, color='green', alpha=0.2)
        # Mark the peak
        plt.plot(p.peak_index, raw_data[p.peak_index], "rx", markersize=10)
        # Label
        plt.text(p.peak_index, raw_data[p.peak_index] + 1, f"P{i+1}", 
                 ha='center', color='black', fontweight='bold')

    plt.title(f"Refactored Pulse Finder Test\nFound {len(found_pulses)} pulses (Should be 3)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()