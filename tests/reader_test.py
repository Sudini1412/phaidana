import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import List, Tuple, Optional
from tqdm import tqdm

# Phaidana imports
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from phaidana.analysis.filter import Filter
from phaidana.analysis.fits import Fit
import phaidana.analysis.pulse_finder as pf

# --- CONSTANTS & CONFIGURATION ---
SAMPLE_RATE_HZ = 125e6
ADC_BITS = 16
VOLTAGE_RANGE = 4.0
ADC2V = VOLTAGE_RANGE / (2**ADC_BITS)
channel_idx = 0

#Pulse quality
tau2_vals = []
fastArea = 80 #640 ns
purityVal = 1.8

# Thresholds and Gates
SATURATION_THRESHOLD_ADC = 55000
PRE_TRIGGER_THRESHOLD_ADC = 200
BASELINE_GATE = 250
PRE_TRIGGER_GATE = 250

# Lower bound for number of pulses per event
npulses = 1

# Number of event plots
nplots = 5

# Paths
DATA_DIR = "/bundle/data/DarkSide/phaidaq/run01915"
OUTPUT_PATH = "/user/sudini/Developer/Data_phaidana/df_1915_Co60_data.csv"


def get_time_axis(num_samples: int) -> np.ndarray:
    """Generates a time axis in microseconds."""
    # (samples / rate) * 1e6 for microseconds
    return (np.arange(1, num_samples + 1, dtype=float) / SAMPLE_RATE_HZ) * 1e6


def visualize_event(waveform: np.ndarray, 
                    time: np.ndarray, 
                    filtered_trace: np.ndarray, 
                    pulses: list, 
                    config: pf.PulseConfig, 
                    event_id: int, 
                    channel_id: int,
                    ExpoFitVals: np.ndarray, 
                    errors: np.ndarray, 
                    save_path: Optional[str] = None):
    """
    Plots the raw signal, filtered trace, thresholds, and detected pulses.
    """
    fit = Fit()

    plt.figure(figsize=(10, 5))
    
    # 1. Plot Traces
    plt.plot(time, waveform, color='k', alpha=0.5, label='Raw Signal')
    plt.plot(time, filtered_trace, color='blue', linewidth=1.0, label='Filtered Trace')
    
    # 2. Plot Thresholds
    plt.axhline(config.high_threshold, color='red', linestyle='--', alpha=0.5, label=f'High Thresh ({config.high_threshold})')
    plt.axhline(config.low_threshold, color='orange', linestyle='--', alpha=0.5, label=f'Low Thresh ({config.low_threshold})')

    # 3. Highlight Pulses
    for i, p in enumerate(pulses):
        # Highlight the duration of the pulse
        # Ensure indices are within bounds
        t_start = time[p.t_start] if p.t_start < len(time) else time[-1]
        t_end = time[p.t_end] if p.t_end < len(time) else time[-1]
        
        plt.axvspan(t_start, t_end, color='green', alpha=0.2)
        
        # Mark the peak
        if p.peak_index < len(waveform):
            fitted_curve = fit.expo(time[p.peak_index:p.t_end], *ExpoFitVals)
            peak_amp = waveform[p.peak_index] 
            plt.plot(time[p.peak_index:p.t_end], fitted_curve, 'r-', linewidth=2, label='Fit')
            plt.plot(time[p.peak_index], peak_amp, "rx", markersize=10)
            plt.text(time[p.peak_index], peak_amp, f"P{i+1}", 
                     ha='center', va='bottom', color='black', fontweight='bold')

    # 4. Labels and Styling
    plt.title(f"Event {event_id} | Channel {channel_id} | Found {len(pulses)} pulses")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude (V)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() 
    else:
        plt.show()


###---MAIN---###

# 1. Setup Analyzers
signal_filter = Filter()
fit = Fit()
mreader = unpack.MIDASreader(DATA_DIR)

# Setup Pulse Finder Config once (Move out of loop for performance)
pf_config = pf.PulseConfig(
    high_threshold=0.012, 
    low_threshold=0.0,
    avg_gate=10,
    min_gap_samples=50,
    min_after_peak=50,
    valley_depth_threshold=5.0,
    valley_rel_fraction=0.5,
    fprompt=fastArea
)
finder = pf.PulseFinder(pf_config)

pulse_rows = []
count = 0

print("Starting event loop...")

for event_idx, event in enumerate(tqdm(mreader)):
    # Basic Validation
    if len(event.adc_data) == 0:
        continue

    wfs = event.adc_data # shape = (n_channels, n_samples)
    
    # 2. Pre-processing (Baseline Subtraction & Calibration)
    baseline = signal_filter.get_baseline(wfs, gate=BASELINE_GATE, start=0)
    
    # Convert to Volts immediately after baseline subtraction
    # (wfs - baseline) * scalar is faster than wfs*scalar - baseline*scalar
    wfs_sub = (wfs - baseline) * ADC2V

    # 3. Quality Checks (Saturation & Pre-trigger pulses)
    # Check max amplitude of the specific channel we are interested in
    is_saturated = np.max(wfs_sub[channel_idx]) > (SATURATION_THRESHOLD_ADC * ADC2V)
    
    # Check for pre-trigger pileup
    # Note: pulse_in_pretrig usually returns a list/array
    pre_trigger_pulses = signal_filter.pulse_in_pretrig(
        wfs_sub, 
        threshold=PRE_TRIGGER_THRESHOLD_ADC * ADC2V,
        gate=PRE_TRIGGER_GATE, 
        start=0
    )
    has_pretrigger = len(pre_trigger_pulses) > 0

    # Skip if signal is bad
    if is_saturated or has_pretrigger:
        continue

    # 4. Pulse Finding
    # Generate time axis only when needed (or cache it if length is constant)
    time_axis = get_time_axis(len(wfs_sub[channel_idx]))
    
    pulses, filtered_trace = finder.process(wfs_sub[channel_idx])

    if len(pulses) == npulses:
        # Select and Store Data
        for p in pulses:
            try:
                # Fitting an exponential to the data post first pulse index and before pulse end
                ExpoFitVals, errors = fit.fit_expo(time_axis[p.peak_index:p.t_end], wfs_sub[channel_idx][p.peak_index:p.t_end], p0 = (p.peak_amplitude, 1.8))
                #if ExpoFitVals is not None:
                    #tau2_vals.append(ExpoFitVals[1]) # Append the decay constant to the tau2 array 
                if ExpoFitVals is not None and ExpoFitVals[1] > purityVal: # If the pulse crosses the baseline AND the decay constant is greater than the set val AND the fit is reasonable
                    count += 1
                    # Progress logging
                    if count % 5000 == 0:
                        print(f"Events with pulses found: {count}")

                    row = asdict(p)
                    row['event_number'] = event_idx
                    row['channel'] = channel_idx
                    row['tau2'] = ExpoFitVals[1]
                    pulse_rows.append(row)

                    # Plotting (First 60 valid events)
                    if count <= nplots:
                        print(f"Plotting Event {event_idx}...")
                        visualize_event(
                            waveform=wfs_sub[channel_idx],
                            time=time_axis,
                            filtered_trace=filtered_trace,
                            pulses=pulses,
                            config=pf_config,
                            event_id=event_idx,
                            channel_id=channel_idx,
                            ExpoFitVals = ExpoFitVals, 
                            errors = errors
                            )
            except:
                pass

print(f"Loop finished. Total events with pulses: {count}")

# 5. Export Data
if pulse_rows:
    print("Creating DataFrame...")
    pulse_df = pd.DataFrame(pulse_rows)
    
    # Reorder columns: event_number, channel first and tau2 first
    cols = ['event_number', 'channel', 'tau2'] + [c for c in pulse_df.columns if c not in ['event_number', 'channel', 'tau2']]
    pulse_df = pulse_df[cols]
    
    print(f"Saving to {OUTPUT_PATH}...")
    pulse_df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")
else:
    print("No pulses found in dataset.")
