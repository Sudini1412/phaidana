import numpy as np 
import sys
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from scipy.signal import find_peaks
from phaidana.analysis.filter import Filter
import phaidana.analysis.pulse_finder as pf
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict


# --- HELPER FUNCTION: PLOTTING ---
def visualize_event(waveform, filtered_trace, pulses, config, event_id, channel_id, save_path=None):
    """
    Plots the raw signal, filtered trace, thresholds, and detected pulses.
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Traces
    plt.plot(waveform, color='k', alpha=0.5, label='Raw Signal')
    plt.plot(filtered_trace, color='blue', linewidth=1.0, label='Filtered Trace')
    
    # 2. Plot Thresholds
    plt.axhline(config.high_threshold, color='red', linestyle='--', alpha=0.5, label=f'High Thresh ({config.high_threshold})')
    plt.axhline(config.low_threshold, color='orange', linestyle='--', alpha=0.5, label=f'Low Thresh ({config.low_threshold})')

    # 3. Highlight Pulses
    for i, p in enumerate(pulses):
        # Highlight the duration of the pulse
        plt.axvspan(p.t_start, p.t_end, color='green', alpha=0.2)
        
        # Mark the peak
        # Note: We access the amplitude from the waveform using the index
        peak_amp = waveform[p.peak_index] 
        plt.plot(p.peak_index, peak_amp, "rx", markersize=10)
        plt.text(p.peak_index, peak_amp, f"P{i+1}", 
                 ha='center', va='bottom', color='black', fontweight='bold')

    # 4. Labels and Styling
    plt.title(f"Event {event_id} | Channel {channel_id} | Found {len(pulses)} pulses")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Amplitude (ADC)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close memory to prevent leaks
    else:
        plt.show()


# --- MAIN SCRIPT --- #
filter = Filter()
dir = "/bundle/data/DarkSide/phaidaq/run01906"
file_path_source = '/user/sudini/Developer/Data_phaidana/df_1906_Co60_data.csv'

source_df = pd.read_csv(file_path_source)
target_channel = 0 
subset_source = source_df[source_df['channel'] == target_channel]

plt.hist(subset_source['tau2'].to_numpy(),bins=200, range=(0,3))
plt.show()
# Define the range of integrals you are interested in (e.g., based on the plot)
min_integral = 50 
max_integral = 250
min_fprompt = 0.02
max_fprompt = 0.14
# Apply the filter
# We use & for "AND", and enclose conditions in parentheses ()
events_in_range = subset_source[
    (subset_source['integral'] >= min_integral) & 
    (subset_source['integral'] <= max_integral) &
    (subset_source['prompt_fraction'] <= min_fprompt) &
    (subset_source['prompt_fraction'] <= max_fprompt)
]

print(f"\n--- Analysis for Integral range [{min_integral}, {max_integral}] ---")
print(f"Found {len(events_in_range)} events.")

if not events_in_range.empty:
    print("Event Numbers:")
    # .unique() ensures we don't double count if an event has multiple pulses in that range
    # (though your logic earlier seemed to only save single-pulse events)
    print(events_in_range['event_number'].unique())
    print(events_in_range['tau2'].to_numpy())
    plt.hist(events_in_range['tau2'].to_numpy(), bins=200, range=(0,3))
    plt.show()
    
    # Optional: Print full details of the first 5 matches
    print("\nFirst 5 matching rows:")
    print(events_in_range.head(5))

    count = 0
    pulse_rows = []
    mreader= unpack.MIDASreader(dir)
    main_events = events_in_range['event_number'].unique()


    print("Starting event loop...")
    for i,event in enumerate(mreader):
        nevent = i
        ch_idx = 0
        
        event_channels = event.nchannels
        # print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
        # print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
        # access waveforms in event
        wfs = event.adc_data # shape = (number of channels, number of waveform sample)
        bal = filter.get_baseline(wfs, gate=250, start=0)
        wfs_sub = wfs - bal
        bal_rms = np.sqrt(np.mean(bal**2))
        bal_std = np.std(bal)

        if nevent in main_events:
            plt.plot(wfs_sub[ch_idx])
            plt.title(f'Plotting event {nevent}')
            plt.show()
