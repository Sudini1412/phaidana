import phaidana.parser.pyreader.VX274X_unpacker as unpack
import matplotlib.pyplot as plt

dir = "/bundle/data/DarkSide/phaidaq/run01942"

mreader= unpack.MIDASreader(dir)
for i,event in enumerate(mreader):
    # print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
    # print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
    # access waveforms in event
    wfs = event.adc_data # shape = (number of channels, number of waveform sample)
    
    if len(wfs)==0:
        pass
    else:
        plt.plot(wfs[0])
        plt.show()