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

# --- CONFIGURATION ---
CONFIG = {
    'sample_rate_hz': 125e6,
    'adc_bits': 16,
    'voltage_range': 4.0,
    'channel_idx': 0,
    'saturation_adc': 55000,
    'pre_trigger_adc': 200,
    'baseline_gate': 250,
    'pre_trigger_gate': 250,
    'n_pulses_req': 1,
    'n_plots': 5,
    'data_dir': "/bundle/data/DarkSide/phaidaq/run01905",
    'output_path': "/user/sudini/Developer/Data_phaidana/df_1905_background_data.csv"
}

# Derived Constants
ADC2V = CONFIG['voltage_range'] / (2**CONFIG['adc_bits'])
SATURATION_V = CONFIG['saturation_adc'] * ADC2V
PRE_TRIGGER_V = CONFIG['pre_trigger_adc'] * ADC2V

# Analysis Parameters
FAST_AREA = 80       # fprompt window
PURITY_VAL = 0.4     # Minimum decay constant
MAX_FIT_ERROR = 1.0  # Maximum allowed error on decay constant

def get_time_axis(num_samples: int, rate_hz: float) -> np.ndarray:
    """Generates a time axis in microseconds."""
    return (np.arange(num_samples, dtype=float) / rate_hz) * 1e6

def visualize_event(time: np.ndarray, 
                    waveform: np.ndarray, 
                    filtered_trace: np.ndarray, 
                    pulses: list, 
                    pf_config: pf.PulseConfig, 
                    fit_engine: Fit,
                    fit_vals: np.ndarray, 
                    event_id: int, 
                    channel_id: int):
    """
    Plots the raw signal, filtered trace, thresholds, and detected pulses.
    """
    plt.figure(figsize=(10, 5))
    
    # 1. Traces
    plt.plot(time, waveform, color='k', alpha=0.5, label='Raw Signal')
    plt.plot(time, filtered_trace, color='blue', linewidth=1.0, label='Filtered')
    
    # 2. Thresholds
    plt.axhline(pf_config.high_threshold, color='red', ls='--', alpha=0.5, label='High Thresh')
    plt.axhline(pf_config.low_threshold, color='orange', ls='--', alpha=0.5, label='Low Thresh')

    # 3. Pulses & Fits
    for i, p in enumerate(pulses):
        # Clip indices to prevent out-of-bounds errors during plotting
        t0_idx = min(p.t_start, len(time) - 1)
        t1_idx = min(p.t_end, len(time) - 1)
        pk_idx = min(p.peak_index, len(time) - 1)

        # Highlight area
        plt.axvspan(time[t0_idx], time[t1_idx], color='green', alpha=0.2)
        
        # Plot Fit
        if fit_vals is not None and pk_idx < t1_idx:
            # Generate curve for the range [peak, end]
            t_slice = time[pk_idx:t1_idx+1] # +1 for inclusive
            if t_slice.size > 0:
                # Note: Assuming fit.expo takes (t, *params)
                fitted_curve = fit_engine.expo(t_slice, *fit_vals)
                plt.plot(t_slice, fitted_curve, 'r-', linewidth=2, label='Exp Fit' if i==0 else "")

        # Mark Peak
        plt.plot(time[pk_idx], waveform[pk_idx], "rx", markersize=10)
        plt.text(time[pk_idx], waveform[pk_idx], f"P{i+1}", 
                 ha='center', va='bottom', color='black', fontweight='bold')

    plt.title(f"Event {event_id} | Channel {channel_id} | Found {len(pulses)} pulses")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude (V)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Setup Analyzers
    signal_filter = Filter()
    fitter = Fit() # Instantiate once
    mreader = unpack.MIDASreader(CONFIG['data_dir'])

    # Pulse Finder Configuration
    pf_config = pf.PulseConfig(
        high_threshold=0.012, 
        low_threshold=0.0,
        avg_gate=10,
        min_gap_samples=250,
        min_after_peak=250,
        valley_depth_threshold=0.05,
        valley_rel_fraction=0.5,
        fprompt=FAST_AREA
    )
    finder = pf.PulseFinder(pf_config)

    pulse_rows = []
    events_with_pulses = 0
    time_axis = None # Cache for time axis

    print("Starting event loop...")

    # Iterate
    for event_idx, event in enumerate(tqdm(mreader)):
        if len(event.adc_data) == 0:
            continue

        wfs = event.adc_data # (n_channels, n_samples)
        
        # 2. Pre-processing
        # Baseline subtraction (Vectorized)
        baseline = signal_filter.get_baseline(wfs, gate=CONFIG['baseline_gate'], start=0)
        
        # Convert to Volts (Vectorized)
        # (wfs - baseline) * scalar is efficient
        wfs_sub = (wfs - baseline) * ADC2V

        # Extract the specific channel of interest once
        channel_wf = wfs_sub[CONFIG['channel_idx']]

        # 3. Quality Checks (Fail Fast)
        # Check Saturation
        if np.max(channel_wf) > SATURATION_V:
            continue
        
        # Check Pre-trigger pileup
        # Note: If pulse_in_pretrig is slow, ensure we only run it on the specific channel if possible. 
        # Assuming current implementation requires full array 'wfs_sub'.
        pre_trigger_pulses = signal_filter.pulse_in_pretrig(
            wfs_sub, 
            threshold=PRE_TRIGGER_V,
            gate=CONFIG['pre_trigger_gate'], 
            start=0
        )
        if len(pre_trigger_pulses) > 0:
            continue

        # 4. Generate Time Axis (Once)
        if time_axis is None or len(time_axis) != len(channel_wf):
            time_axis = get_time_axis(len(channel_wf), CONFIG['sample_rate_hz'])

        # 5. Pulse Finding
        pulses, filtered_trace = finder.process(channel_wf)

        # Logic: Only proceed if exact number of pulses found
        if len(pulses) == CONFIG['n_pulses_req']:
            
            for p in pulses:
                # Sanity check indices
                if p.peak_index >= p.t_end: continue

                try:
                    # Slice data for fitting
                    t_slice = time_axis[p.peak_index : p.t_end]
                    y_slice = channel_wf[p.peak_index : p.t_end]

                    # Fit Exponential
                    # p0 = (Amplitude, Decay constant guess)
                    expo_vals, errors = fitter.fit_expo(t_slice, y_slice, p0=(p.peak_amplitude, 1.8))
                    
                    # Validation of Fit
                    # Check if fit succeeded (expo_vals not None)
                    # Check decay constant (expo_vals[1]) > threshold
                    # Check error is small
                    if (expo_vals is not None and 
                        expo_vals[1] > PURITY_VAL and 
                        errors[1] < MAX_FIT_ERROR):
                        
                        events_with_pulses += 1
                        
                        # Logging
                        if events_with_pulses % 5000 == 0:
                            print(f"Events with valid pulses: {events_with_pulses}")

                        # Store Data
                        row = asdict(p)
                        row.update({
                            'event_number': event_idx,
                            'channel': CONFIG['channel_idx'],
                            'tau2': expo_vals[1]
                        })
                        pulse_rows.append(row)

                        # Visualization (Early events only)
                        if events_with_pulses <= CONFIG['n_plots']:
                            print(f"Plotting Event {event_idx}...")
                            visualize_event(
                                time=time_axis,
                                waveform=channel_wf,
                                filtered_trace=filtered_trace,
                                pulses=pulses,
                                pf_config=pf_config,
                                fit_engine=fitter,
                                fit_vals=expo_vals,
                                event_id=event_idx,
                                channel_id=CONFIG['channel_idx']
                            )

                except (RuntimeError, ValueError):
                    # Fit failed to converge
                    continue
                except Exception as e:
                    print(f"Unexpected error in Event {event_idx}: {e}")
                    continue

    print(f"Loop finished. Total valid events: {events_with_pulses}")

    # 6. Export
    if pulse_rows:
        print("Creating DataFrame...")
        pulse_df = pd.DataFrame(pulse_rows)
        
        # Organize Columns
        priority_cols = ['event_number', 'channel', 'tau2']
        cols = priority_cols + [c for c in pulse_df.columns if c not in priority_cols]
        pulse_df = pulse_df[cols]
        
        print(f"Saving to {CONFIG['output_path']}...")
        pulse_df.to_csv(CONFIG['output_path'], index=False)
        print("Done.")
    else:
        print("No valid pulses found in dataset.")

if __name__ == "__main__":
    main()