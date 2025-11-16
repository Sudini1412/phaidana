import numpy as np 
import phaidana.parser.pyreader.VX274X_unpacker as unpack
from scipy.signal import find_peaks
from phaidana.analysis.filter import Filter
from phaidana.analysis.pulse_finder import Pulse

import matplotlib.pyplot as plt

filter = Filter()
pulse = Pulse()

dir = "/bundle/data/DarkSide/phaidaq/run01906"

mreader= unpack.MIDASreader(dir)
for i,event in enumerate(mreader):
    # print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
    # print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
    # access waveforms in event
    wfs = event.adc_data # shape = (number of channels, number of waveform sample)
    bal = filter.get_baseline(wfs, gate=250, start=0)
    bal_rms = np.rms(bal)
    wfs_sub = wfs - bal
    if len(wfs)==0:
        pass
    else:
        if np.max(wfs[0]) > 55000:
            pass
        else:
            peaks, properties = find_peaks(wfs_sub[0], height=1000)
            plt.plot(wfs_sub[0]) 
            plt.plot(peaks, wfs_sub[0][peaks], 'ro')
            plt.show()           

