import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================

file_path_source = "/user/sudini/Developer/Data_phaidana/df_1906_12mv_roi350_pf113_t0_Co60_data.csv"
file_path_background = "/user/sudini/Developer/Data_phaidana/df_1905_12mv_roi350_pf113_t0_background_data.csv"
# --- DEFINE YOUR FILTERS HERE ---
# Syntax for ranges: (min_value, max_value). Use None for no limit.
# Syntax for exact match: value
FILTER_CONFIG = {
    'channel': 0,                # Exact match: Channel must be 0
    #'integral': (0, None),       # Range: Must be > 0 (None means infinity)
    'tau2': (0.0, None),         # Range: Must be > 0.0
    'tau2_err': (None, 3),       # Range: Must be < 3
    #'peak_amplitude': (0.08,None), # Range: Must be > 0.1
    'prompt_fraction': (0, 1),
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def apply_custom_filters(df, filters):
    """
    Filters a dataframe based on a dictionary of conditions.
    """
    df_filtered = df.copy()
    
    print(f"--- Filtering Data ({len(df)} initial events) ---")
    
    for col, condition in filters.items():
        # Check if column exists to avoid errors
        if col not in df_filtered.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Case 1: Condition is a tuple (Range: min, max)
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            min_val, max_val = condition
            
            if min_val is not None:
                df_filtered = df_filtered[df_filtered[col] > min_val]
            if max_val is not None:
                df_filtered = df_filtered[df_filtered[col] < max_val]
                
        # Case 2: Condition is a single value (Exact match)
        else:
            df_filtered = df_filtered[df_filtered[col] == condition]
            
    print(f"-> Remaining events: {len(df_filtered)}")
    return df_filtered

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================

# Load Data
print("Loading CSV files...")
source_df = pd.read_csv(file_path_source)
background_df = pd.read_csv(file_path_background)

# Apply Filters
print("\nProcessing Source Data:")
subset_source = apply_custom_filters(source_df, FILTER_CONFIG)

print("\nProcessing Background Data:")
subset_background = apply_custom_filters(background_df, FILTER_CONFIG)

# ==========================================
# 4. PLOTTING
# ==========================================

target_channel = FILTER_CONFIG.get('channel', 'All')

# --- Plot 1: Integral Histogram ---
plt.figure(figsize=(10, 6))
plt.hist(subset_source['integral'], bins=200, histtype='step', range=(0, 200), 
         color='black', label=f'Source Channel {target_channel}')
plt.hist(subset_background['integral'], bins=200, histtype='step', range=(0, 200), 
         color='red', label=f'Background Channel {target_channel}')

plt.title(f"Pulse Integral Spectrum - Channel {target_channel}")
plt.xlabel("Integral (Volts x Samples)")
plt.ylabel("Counts")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot 2: Scatter (Peak Index vs Amplitude) ---
plt.figure(figsize=(10, 6))
plt.scatter(subset_source['peak_index'], subset_source['peak_amplitude'], s=1, color='k', label='Source')
plt.title("Peak Index vs Amplitude")
plt.xlabel("Peak Index")
plt.ylabel("Peak Amplitude")
plt.grid(True, alpha=0.3)
plt.show()

# --- Plot 3: 2D Hist (Source: Integral vs PF) ---
plt.figure(figsize=(8, 6))
plt.hist2d(subset_source['integral'], subset_source['prompt_fraction'], bins=200, cmin=1, cmap='viridis')
plt.colorbar(label='Counts')
plt.title(f"Source: Prompt Fraction vs Integral (Ch {target_channel})")
plt.xlabel("Integral")
plt.ylabel("Prompt Fraction")
plt.show()

# --- Plot 4: 2D Hist (Background: Integral vs PF) ---
plt.figure(figsize=(8, 6))
plt.hist2d(subset_background['integral'], subset_background['prompt_fraction'], bins=200, cmin=1, cmap='inferno')
plt.colorbar(label='Counts')
plt.title(f"Background: Prompt Fraction vs Integral (Ch {target_channel})")
plt.xlabel("Integral")
plt.ylabel("Prompt Fraction")
plt.show()

# ==========================================
# 5. DATA INSPECTION
# ==========================================

# Inspect specific anomalies (e.g., peak_index == 0)
# We can use the logic from the loop you provided, but faster using pandas filtering
print("\n--- Events with Peak Index == 0 ---")
anomalies = subset_source[subset_source['peak_index'] == 0]

if not anomalies.empty:
    print(anomalies[['peak_index', 'prompt_fraction', 'event_number']].to_string(index=False))
else:
    print("No events found with peak_index == 0 in the current selection.")