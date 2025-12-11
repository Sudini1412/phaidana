import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use the same path you defined in reader.py
file_path_source = '/user/sudini/Developer/Data_phaidana/df_1915_pf640_t0_Co60_data.csv'
file_path_background =  '/user/sudini/Developer/Data_phaidana/df_1916_background_data.csv'

source_df = pd.read_csv(file_path_source)
background_df = pd.read_csv(file_path_background)
print(source_df)
target_channel = 0 
subset_source = source_df[(source_df['channel'] == target_channel) & (source_df['integral'] > 0) & (source_df['prompt_fraction'] > 0) & (source_df['prompt_fraction'] < 0.5)]
subset_background = background_df[(background_df['channel'] == target_channel) & (background_df['integral'] > 0) & (background_df['prompt_fraction'] > 0) & (background_df['prompt_fraction'] < 0.5)]
# Check if we actually have data for this channel
print(f"Plotting {len(subset_source)} pulses for Channel {target_channel}")

# 3. Plot the Histogram
plt.figure(figsize=(10, 6))

# 'bins' determines the resolution. Increase it for finer detail.
# 'log=True' helps see small features if you have a large range of counts.
plt.hist(subset_source['integral'], bins=200, histtype='step', range=(0,200), color='black', alpha=0.7, log=False, label=f'Source Channel {target_channel}')
plt.hist(subset_background['integral'], bins=200, range =(0,200), histtype='step', color='r', alpha=0.7, log=False, label=f'Background Channel {target_channel}')

plt.title(f"Pulse Integral Spectrum - Channel {target_channel}")
plt.xlabel("Integral (Volts x Samples)")
plt.ylabel("Counts")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

#plt.savefig('/user/sudini/Developer/Plots/charge_spectrum.png')
plt.show()

plt.scatter(subset_source['peak_index'].to_numpy(),subset_source['peak_amplitude'].to_numpy(), s=1, color='k')
#plt.savefig('/user/sudini/Developer/Plots/amps.png')
plt.show()

plt.hist2d(subset_source['integral'].to_numpy(),subset_source['prompt_fraction'].to_numpy(), bins=150, cmin = 1)
plt.colorbar()
#plt.savefig('/user/sudini/Developer/Plots/prompt_fraction.png')
plt.show()
# --- Retrieve Events in Range ---


for pk,amp,ev in zip(subset_source['peak_index'],subset_source['prompt_fraction'],subset_source['event_number']):
    if pk==0:
        print(pk,amp,ev)
