import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use the same path you defined in reader.py
file_path_source = '/user/sudini/Developer/Data_phaidana/df_1915_Co60_data.csv'
#file_path_background =  '/user/sudini/Developer/Data_phaidana/df_1916_background_data.csv'

#file_path_source = '/user/sudini/Developer/Data_phaidana/cannoli_df_1915_Co60_data.csv'
#file_path_background = '/user/sudini/Developer/df_1915_Co60_50ppmDope_data.csv'

source_df = pd.read_csv(file_path_source)
#background_df = pd.read_csv(file_path_background)

target_channel = 0 
subset_source = source_df[source_df['channel'] == target_channel]
#subset_background = background_df[background_df['channel'] == target_channel]
# Check if we actually have data for this channel
print(f"Plotting {len(subset_source)} pulses for Channel {target_channel}")

# 3. Plot the Histogram
plt.figure(figsize=(10, 6))

# 'bins' determines the resolution. Increase it for finer detail.
# 'log=True' helps see small features if you have a large range of counts.
plt.hist(subset_source['integral'], bins=300, range =(0,5000000), histtype='step', color='black', alpha=0.7, log=False, label=f'Source Channel {target_channel}')
#plt.hist(subset_background['integral'], bins=300, range =(0,5000000), histtype='step', color='r', alpha=0.7, log=False, label=f'Background Channel {target_channel}')

plt.title(f"Pulse Integral Spectrum - Channel {target_channel}")
plt.xlabel("Integral (ADC x Samples)")
plt.ylabel("Counts")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.show()

plt.scatter(subset_source['peak_index'].to_numpy(),subset_source['peak_amplitude'].to_numpy(), s=1, color='k')
plt.show()

plt.hist2d(subset_source['integral'].to_numpy(),subset_source['prompt_fraction'].to_numpy(), bins=400, cmin = 1, range = ([0,1e7],[0,1]))
plt.colorbar()
plt.show()
# --- Retrieve Events in Range ---

# Define the range of integrals you are interested in (e.g., based on the plot)
min_integral = 1000000 
max_integral = 1500000

# Apply the filter
# We use & for "AND", and enclose conditions in parentheses ()
events_in_range = subset_source[
    (subset_source['integral'] >= min_integral) & 
    (subset_source['integral'] <= max_integral)
]

print(f"\n--- Analysis for Integral range [{min_integral}, {max_integral}] ---")
print(f"Found {len(events_in_range)} events.")

if not events_in_range.empty:
    print("Event Numbers:")
    # .unique() ensures we don't double count if an event has multiple pulses in that range
    # (though your logic earlier seemed to only save single-pulse events)
    print(events_in_range['event_number'].unique())
    
    # Optional: Print full details of the first 5 matches
    print("\nFirst 5 matching rows:")
    print(events_in_range.head(5))