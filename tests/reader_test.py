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
    plt.figure(figsize=(10, 5))
    
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
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close memory to prevent leaks
    else:
        plt.show()


# --- MAIN SCRIPT --- #
filter = Filter()
dir = "/bundle/data/DarkSide/phaidaq/run01915"
count = 0
pulse_rows = []
dig_sample_rate = 125*(10**6)
ADC2V = float(4 / 2**16)
mreader= unpack.MIDASreader(dir)

print("Starting event loop...")
for i,event in enumerate(mreader):
    nevent = i
    event_channels = event.nchannels
    # print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
    # print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
    # access waveforms in event
    wfs = event.adc_data # shape = (number of channels, number of waveform sample)
    bal = filter.get_baseline(wfs, gate=250, start=0)
    wfs_sub = wfs - bal
    bal_rms = np.sqrt(np.mean(bal**2))
    bal_std = np.std(bal)
    wfs_sub = wfs_sub * ADC2V
    print(wfs_sub)



    # Checks for empty events
    if len(wfs)==0:
        continue

    if (np.max(wfs[0]) > 55000 * ADC2V) or (len(filter.pulse_in_pretrig(wfs_sub ,gate=250, start=0))>0):
        pass
    else:
        time = np.arange(1,len(wfs[0])+1) * dig_sample_rate
        wfs_sub = wfs_sub * ADC2V
        # Run Finder
        config = pf.PulseConfig(high_threshold  = 200.0, 
                                low_threshold = 0.0,
                                avg_gate = 10,
                                min_gap_samples = 50,
                                min_after_peak = 50,
                                valley_depth_threshold = 5.0,
                                valley_rel_fraction = 0.5,
                                fprompt = 80)
        finder = pf.PulseFinder(config)

        channel_idx = 0
        pulses,  filtered_trace = finder.process(wfs_sub[channel_idx])

        pulse_tot = 0
    
        if len(pulses) >0:
            count += 1
            if count % 5000 == 0:
                print(f"Pulses found: {count}")
            
            # --- CALL PLOT FUNCTION ---
            # Logic: Only plot the first 5 found pulses to check quality, then stop plotting
            if count <= 60:
                print(f"Plotting Event {nevent}...")
                visualize_event(
                    waveform=wfs_sub[channel_idx],
                    filtered_trace=filtered_trace,
                    pulses=pulses,
                    config=config,
                    event_id=nevent,
                    channel_id=channel_idx
                    # save_path=f"./debug_plot_{nevent}.png" # Uncomment to save instead of show
                )

            for p in pulses:
                row = asdict(p)
                row['event_number'] = nevent
                row['channel'] = channel_idx
                pulse_rows.append(row)
            



print("Final pulse count: ", count)

print("Creating DataFrame...")
pulse_df = pd.DataFrame(pulse_rows)
cols = ['event_number', 'channel'] + [c for c in pulse_df.columns if c not in ['event_number', 'channel']]
pulse_df = pulse_df[cols]
print(pulse_df)

output_path = '/user/sudini/Developer/Data_phaidana/df_1915_Co60_data.csv'
pulse_df.to_csv(output_path, index=False)

