import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Phaidana imports
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from phaidana.analysis.filter import Filter

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================

DATA_DIR = "/bundle/data/DarkSide/phaidaq/run01915"
CSV_PATH = "/user/sudini/Developer/Data_phaidana/df_1915_12mv_pf900_t100_Co60_data.csv"

# Signal Processing Constants
SAMPLING_RATE_HZ = 125_000_000.0  # 125 MHz
ADC_DYNAMIC_RANGE = 4.0  # Volts
ADC_RESOLUTION_BITS = 16
ADC2V = ADC_DYNAMIC_RANGE / (2**ADC_RESOLUTION_BITS)

# --- FILTER SETTINGS ---
# Syntax: 'Column_Name': (Min_Value, Max_Value)
# Use None for no limit (e.g., (0.5, None) means > 0.5)
FILTER_CONFIG = {
    'channel': 0,                     # Exact match
    'integral': (20, 90),            # Range: Integration
    'prompt_fraction': (0.02, 0.1),  # Range: fprompt
    'tau2': (0.0, None),              # Range: tau2 (must be > 0.4)
    'tau2_err': (None, 3.0)           # Range: tau2 error (must be < 3.0)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_time_axis(num_samples: int, rate_hz: float) -> np.ndarray:
    """Generates a time axis in microseconds."""
    return (np.arange(num_samples, dtype=float) / rate_hz) * 1e6

def plot_waveform(time_axis, waveform, event_id, integral, fprompt):
    """
    Plots the waveform with event info in the title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, waveform, linewidth=0.8, color='k')
    
    # Title now shows the specific parameters for this event
    plt.title(f'Event {event_id} | Int: {integral:.1f} | fprompt: {fprompt:.3f}')
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def apply_custom_filters(df, filters):
    """
    Generic function to filter a dataframe based on config dictionary.
    """
    df_filtered = df.copy()
    print(f"--- Filtering Data ({len(df)} initial events) ---")
    
    for col, condition in filters.items():
        if col not in df_filtered.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Case 1: Range Filter (tuple/list)
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            min_val, max_val = condition
            if min_val is not None:
                df_filtered = df_filtered[df_filtered[col] >= min_val]
            if max_val is not None:
                df_filtered = df_filtered[df_filtered[col] <= max_val]
            print(f"  Filter {col}: {condition} -> {len(df_filtered)} events left")

        # Case 2: Exact Match (single value)
        else:
            df_filtered = df_filtered[df_filtered[col] == condition]
            print(f"  Filter {col} == {condition} -> {len(df_filtered)} events left")
            
    return df_filtered

def load_and_filter_data(csv_path, filters):
    """Loads CSV and returns filtered DataFrame."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Apply filters
    filtered_df = apply_custom_filters(df, filters)
    
    # Optional: Plot distributions of the remaining data to confirm filters work
    if not filtered_df.empty:
        plt.figure(figsize=(12, 4))
        
        # Plot Integral
        plt.subplot(1, 3, 1)
        plt.hist(filtered_df['integral'], bins=50, color='skyblue', histtype='stepfilled')
        plt.title('Filtered Integral')
        
        # Plot Prompt Fraction (if exists)
        if 'prompt_fraction' in filtered_df.columns:
            plt.subplot(1, 3, 2)
            plt.hist(filtered_df['prompt_fraction'], bins=50, color='salmon', histtype='stepfilled')
            plt.title('Filtered fprompt')
            
        # Plot Tau2 (if exists)
        if 'tau2' in filtered_df.columns:
            plt.subplot(1, 3, 3)
            plt.hist(filtered_df['tau2'], bins=50, color='lightgreen', histtype='stepfilled')
            plt.title('Filtered Tau2')
            
        plt.tight_layout()
        plt.show()

    return filtered_df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    # 1. Load and Filter CSV Data
    filtered_df = load_and_filter_data(CSV_PATH, FILTER_CONFIG)
    
    if filtered_df.empty:
        print("No events found matching criteria. Exiting.")
        return

    # Create a lookup dictionary: event_number -> {integral, prompt_fraction, etc}
    # This allows us to access the event properties quickly inside the loop
    target_event_map = filtered_df.set_index('event_number')[['integral', 'prompt_fraction']].to_dict('index')
    
    print("\nFirst 5 matching rows:")
    print(filtered_df.head(5))

    # 2. Setup Data Reader
    print(f"\nInitializing Reader for directory: {DATA_DIR}")
    mreader = unpack.MIDASreader(DATA_DIR)
    signal_filter = Filter()
    
    # 3. Process Events
    print("Starting event loop...")
    
    time_axis = None
    target_channel = FILTER_CONFIG.get('channel', 0)
    
    # Iterate through MIDAS file
    for i, event in enumerate(mreader):
        
        # SKIP if this event index is not in our filtered list
        if i not in target_event_map:
            continue

        if len(event.adc_data) == 0:
            continue

        # Extract Waveforms (shape: n_channels, n_samples)
        wfs = event.adc_data 
        
        # Calculate Baseline
        baseline = signal_filter.get_baseline(wfs, gate=250, start=0) 
        
        # Subtract Baseline and Convert to Volts
        wfs_sub = (wfs - baseline) * ADC2V

        # Get specific channel waveform
        current_wf = wfs_sub[target_channel] 
        
        if time_axis is None or len(time_axis) != len(current_wf):
            time_axis = get_time_axis(len(current_wf), SAMPLING_RATE_HZ)

        # Retrieve metadata for the plot title from our lookup map
        event_meta = target_event_map[i]
        
        print(f"Plotting Event {i}: Int={event_meta['integral']:.1f}, PF={event_meta['prompt_fraction']:.2f}")
        plot_waveform(
            time_axis, 
            current_wf, 
            i, 
            event_meta['integral'], 
            event_meta['prompt_fraction']
        )

if __name__ == "__main__":
    main()