import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from tqdm import tqdm
import json
import argparse
import os
import sys

# Phaidana imports
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from phaidana.analysis.filter import Filter
from phaidana.analysis.fits import Fit
import phaidana.analysis.pulse_finder as pf

def load_config(path: str) -> dict:
    """Loads configuration from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, 'r') as f:
        return json.load(f)

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
                    channel_id: int,
                    fprompt_window: int):
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
        plt.axvspan(time[t0_idx], time[t1_idx], color='green', alpha=0.2, label="ROI integration")
        plt.axvspan(time[t0_idx], time[t0_idx+fprompt_window], color ='blue', alpha=0.2, label="fprompt integration")
        
        # Plot Fit (Only if fit_vals were provided)
        if fit_vals is not None and pk_idx < t1_idx:
            t_slice = time[pk_idx:t1_idx+1]
            if t_slice.size > 0:
                fitted_curve = fit_engine.expo(t_slice, *fit_vals)
                plt.plot(t_slice, fitted_curve, 'r-', linewidth=2, label='Exp Fit' if i==0 else "")

        # Mark Peak
        plt.plot(time[pk_idx], waveform[pk_idx], "rx", markersize=10)
        plt.text(time[pk_idx], waveform[pk_idx], f"P{i+1}", 
                 ha='center', va='bottom', color='black', fontweight='bold')

    plt.title(f"Event {event_id} | Channel {channel_id} | Found {len(pulses)} pulses")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude (V)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main(config: dict):
    """
    Main execution function. Receives the configuration dictionary.
    """
    # --- Derived Constants ---
    adc2v = config['voltage_range'] / (2**config['adc_bits'])
    saturation_v = config['saturation_adc'] * adc2v
    pre_trigger_v = config['pre_trigger_adc'] * adc2v
    
    # --- Check Fit Configuration ---
    # Default to True if key is missing to maintain backward compatibility
    ENABLE_FIT = config['fit_params'].get('enable_fit', True)
    
    # --- Setup Analyzers ---
    signal_filter = Filter()
    fitter = Fit()
    
    print(f"Loading data from: {config['data_dir']}")
    print(f"Tau2 Fit Enabled: {ENABLE_FIT}")
    mreader = unpack.MIDASreader(config['data_dir'])

    # --- Pulse Finder Configuration ---
    pf_params = config['pulse_finder']
    pf_config = pf.PulseConfig(
        high_threshold=pf_params['high_threshold'], 
        low_threshold=pf_params['low_threshold'],
        avg_gate=pf_params['avg_gate'],
        min_gap_samples=pf_params['min_gap_samples'],
        min_after_peak=pf_params['min_after_peak'],
        valley_depth_threshold=pf_params['valley_depth_threshold'],
        valley_rel_fraction=pf_params['valley_rel_fraction'],
        fprompt=pf_params['fprompt_window']  
    )
    finder = pf.PulseFinder(pf_config)

    pulse_rows = []
    events_with_pulses = 0
    time_axis = None 

    max_events = config.get('max_events')
    if max_events == -1: 
        max_events = None

    print("Starting event loop...")

    for event_idx, event in enumerate(tqdm(mreader,total=max_events)):
        if len(event.adc_data) == 0:
            continue

        if max_events is not None and event_idx >= max_events:
            print(f"\nStop requested: Reached limit of {max_events} events.")
            break

        wfs = event.adc_data
        
        # Baseline subtraction & Calibration
        baseline = signal_filter.get_baseline(wfs, gate=config['baseline_gate'], start=0)
        wfs_sub = (wfs - baseline) * adc2v

        # Pre-trigger check
        is_clean_mask = signal_filter.get_pretrig_mask(
            wfs_sub, 
            threshold=pre_trigger_v, 
            gate=config['pre_trigger_gate'], 
            start=0
        )

        if not is_clean_mask[config['channel_idx']]:
            continue

        # Extract specific channel
        channel_wf = wfs_sub[config['channel_idx']]

        # Quality Checks
        if np.max(channel_wf) > saturation_v:
            continue
        
        pre_trigger_pulses = signal_filter.pulse_in_pretrig(
            wfs_sub, 
            threshold=pre_trigger_v,
            gate=config['pre_trigger_gate'], 
            start=0
        )
        if len(pre_trigger_pulses) > 0:
            continue

        # Generate Time Axis (Once)
        if time_axis is None or len(time_axis) != len(channel_wf):
            time_axis = get_time_axis(len(channel_wf), config['sample_rate_hz'])

        # Pulse Finding
        pulses, filtered_trace = finder.process(channel_wf)

        if len(pulses) == config['n_pulses_req']:
            
            for p in pulses:
                if p.t_start < config['pre_trigger_gate']:
                    continue
                if p.peak_index >= p.t_end: continue

                try:
                    # Initialize placeholders for fit results
                    tau2_val = -1.0
                    tau2_err = -1.0
                    fit_vals_for_plot = None
                    pulse_accepted = False

                    # ---------------- FIT LOGIC ----------------
                    if ENABLE_FIT:
                        t_slice = time_axis[p.peak_index : p.t_end]
                        y_slice = channel_wf[p.peak_index : p.t_end]
                        
                        expo_vals, errors = fitter.fit_expo(t_slice, y_slice, p0=(p.peak_amplitude, 1.8))
                        
                        # Filter by Fit Quality
                        if (expo_vals is not None and 
                            expo_vals[1] > config['fit_params']['purity_val'] and 
                            errors[1] < config['fit_params']['max_fit_error']):
                            
                            tau2_val = expo_vals[1]
                            tau2_err = errors[1]
                            fit_vals_for_plot = expo_vals
                            pulse_accepted = True
                    else:
                        # Fit Disabled: Accept pulse automatically based on finder
                        pulse_accepted = True
                        # tau2 values remain -1.0
                    # -------------------------------------------

                    if pulse_accepted:
                        events_with_pulses += 1
                        
                        if events_with_pulses % 5000 == 0:
                            print(f"Events with valid pulses: {events_with_pulses}")

                        row = asdict(p)
                        row.update({
                            'event_number': event_idx,
                            'channel': config['channel_idx'],
                            'tau2': tau2_val,
                            'tau2_err': tau2_err
                        })
                        pulse_rows.append(row)

                        if events_with_pulses <= config['n_plots']:
                            print(f"Plotting Event {event_idx}...")
                            visualize_event(
                                time=time_axis,
                                waveform=channel_wf,
                                filtered_trace=filtered_trace,
                                pulses=pulses,
                                pf_config=pf_config,
                                fit_engine=fitter,
                                fit_vals=fit_vals_for_plot,
                                event_id=event_idx,
                                channel_id=config['channel_idx'],
                                fprompt_window=pf_params['fprompt_window']
                            )

                except (RuntimeError, ValueError):
                    continue
                except Exception as e:
                    print(f"Error in Event {event_idx}: {e}")
                    continue

    print(f"Loop finished. Total valid events: {events_with_pulses}")

    if pulse_rows:
        print("Creating DataFrame...")
        pulse_df = pd.DataFrame(pulse_rows)
        
        priority_cols = ['event_number', 'channel', 'tau2', 'tau2_err']
        cols = priority_cols + [c for c in pulse_df.columns if c not in priority_cols]
        pulse_df = pulse_df[cols]
        
        print(f"Saving to {config['output_path']}...")
        pulse_df.to_csv(config['output_path'], index=False)
        print("Done.")
    else:
        print("No valid pulses found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phaidana Pulse Analysis")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.abspath(os.path.join(script_dir, "..", "config", "config.json"))
    
    parser.add_argument("-c", "--config", type=str, default=default_config_path, 
                        help=f"Path to JSON config (default: {default_config_path})")
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.config):
             print(f"Error: Config file not found at: {args.config}")
        else:
            config_data = load_config(args.config)
            main(config_data)
    except Exception as e:
        print(f"Fatal Error: {e}")