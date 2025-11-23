import numpy as np 
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from scipy.signal import find_peaks
from phaidana.analysis.filter import Filter
import phaidana.analysis.pulse_finder as pf

import matplotlib.pyplot as plt

filter = Filter()
dir = "/bundle/data/DarkSide/phaidaq/run01906"

mreader= unpack.MIDASreader(dir)
for i,event in enumerate(mreader):
    # print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
    # print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
    # access waveforms in event
    wfs = event.adc_data # shape = (number of channels, number of waveform sample)
    bal = filter.get_baseline(wfs, gate=250, start=0)
    wfs_sub = wfs - bal
    bal_rms = np.sqrt(np.mean(bal**2))
    bal_std = np.std(bal)

    if len(wfs)==0:
        pass
    else:
        if (np.max(wfs[0]) > 55000) or (len(filter.pulse_in_pretrig(wfs_sub ,gate=250, start=0))>0):
            pass
        else:

            # Run Finder
            config = pf.PulseConfig(high_threshold  = 200.0, 
                                    low_threshold = 100.0,
                                    avg_gate = 5,
                                    min_gap_samples = 10,
                                    min_after_peak = 100,
                                    valley_depth_threshold = 2.0,
                                    valley_rel_fraction = 0.2)
            finder = pf.PulseFinder(config)
            pulses,  filtered_trace = finder.process(wfs[0])

            print(pulses)
            first_pulse = pulses[0]
            start = first_pulse.t_start
            end = first_pulse.t_end
            print(start)
            plt.plot(wfs_sub[0],color='k')
            plt.plot(filtered_trace,color='b')
            plt.axvline(x=start, color='r')
            plt.axvline(x=end, color='r')
            plt.show()           

