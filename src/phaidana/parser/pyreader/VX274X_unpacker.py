import midas.file_reader as file_reader
import numpy as np
import os, glob
import errno

class MIDASreader:
    '''
    Iterable object to unpack MIDAS events
    Usage:
    mreader=MIDASreader(<list of MIDAS files>, <str indicating the ADC model>, <list of MIDAS data banks>)
    for i,event in enumerate(mreader):
        print(f'Event # {i}\t#Modules {event.nboards}\t#Channels {event.nchannels}\t#Samples {event.nsamples}')
         print(f'\tchannel mask {bin(event.channel_mask)[2:]}\ttrigger time: {event.trigger_time}')
        # access waveforms in event
        event.adc_data # shape = (number of channels, number of waveform sample)

    '''
    def __init__(self,dir):
        try:
            if os.path.isdir(dir):
                print(f'{dir} is a directory')
                # get list of midas files, exclude odb dumps (*.json)
                # compressed (*.mid.lz4, *.mid.gz) or uncompressed
                self.midas_files  = glob.glob(f'{dir}/*mid*')
                # ensure subruns are processed in chrnonological order
                self.midas_files.sort()
        except TypeError:
            # copy the list of files
            self.midas_files  = dir

        self.subidx=0
        self.__next_subrun__()
        
        self.ADCmodel = 'VX2740'
        self.ADCbanks = "D000"
        self.event_number=0

    def isADCbank(self,current_bank):
        '''
        ensure that the data from ADCs and not from other equipment
        '''
        if current_bank in self.ADCbanks: return True
        else: return False

    def __next__(self):
        ev = self.read()
        if not ev:
            raise StopIteration()
        else:
            return ev

    def __iter__(self):
        return self

    def __next_subrun__(self):
        if self.subidx < len(self.midas_files):
            fname=self.midas_files[self.subidx]
            if os.path.isfile(fname):
                self.mfile = file_reader.MidasFile(fname,use_numpy=True)
                print(f'subrun {self.subidx}: {fname}')
                self.subidx+=1
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)

    def read(self):
        '''
         main function to unpack ADC data
         returns waveform array (number of channels, number of samples)
        '''
        for event in self.mfile:
            if event.header.is_midas_internal_event():
                if event.header.is_eor_event():
                    self.__next_subrun__()
                    return self.read()
                continue

            raw = unpack_VX2740()
            
            for bank_name, bank in event.banks.items():
                if len(bank.data) and self.isADCbank(bank_name):
                    raw.unpack(bank.data)

            raw.midas_event=event.header.serial_number
            return raw
                    

class unpack_ADC:
    '''
    Base class to unpack ADC data
    '''
    def __init__(self,model):
        self.name=model
        self.adc_data=np.array([])
        self.nboards=0
        self.nchannels=0
        self.nsamples=0
        self.midas_event=-1
        self.event_counter=np.uint64(0)
        self.trigger_time=np.uint64(0) # ns
        self.channel_mask=0

    def info(self):
        print(f'''ADC: {self.name}   #Modules: {self.nboards:3d}   #Channels: {self.nchannels:4d}
MIDAS S/N: {self.midas_event:6d}   Trigger Counter: {self.event_counter:6d}   Trigger Timestamp: {self.trigger_time:6d} ns''')



class unpack_VX2740(unpack_ADC):
    def __init__(self):
        super().__init__('VX2740')
            
    def unpack(self,bank_data):
        self.unpackHeader(bank_data[:3])
        if self.format == 0x10:
            self.unpackData(bank_data[3:])
            self.nchannels=self.adc_data.shape[0]
            self.nsamples=self.adc_data.shape[1]
        else:
            print(self.name,'no scope data')
        
    def unpackHeader(self,head_data):
        self.header=np.array(head_data,dtype='uint64')
        self.format = self.header[0] >> np.uint64(56)
        self.event_counter = (self.header[0] >> np.uint64(32)) & np.uint64(0xffffff)
        self.event_size = self.header[0] & np.uint64(0xffffffff)
        self.flags = self.header[1] >> np.uint64(52)
        self.overlap = (self.header[1] >> np.uint64(48)) & np.uint64(0xf)
        self.channel_mask = self.header[2]
        self.trigger_time = (self.header[1] & np.uint64(0xffffffffffff))*np.uint64(8)
       
    def unpackData(self,bank_data):
        '''
        data format:
        64bit word channel 0 (4 samples), 64bit word channel 1 (4 samples), 
        ... 64bit word channel N (4 samples)
        '''
        data = np.array(bank_data,dtype='uint64')
        self.nboards+=1

        nchans = bin(self.channel_mask).count('1')
        # only 1/4 of the samples
        nsamples = np.uint32((self.event_size - 3) / nchans)
 
        bitmask=np.uint64(0xffff)
        slot1 = (data&bitmask).reshape(nchans,nsamples,order='F')
        slot2 = ((data>>16)&bitmask).reshape(nchans,nsamples,order='F')
        slot3 = ((data>>32)&bitmask).reshape(nchans,nsamples,order='F')
        slot4 = ((data>>48)&bitmask).reshape(nchans,nsamples,order='F')
        waveforms = np.vstack((slot1,slot2,slot3,slot4)).reshape(nchans,nsamples*4,order='F')
        self.adc_data=np.vstack([self.adc_data,waveforms]) if self.adc_data.size else waveforms




    