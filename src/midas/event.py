"""
Represents data from a MIDAS-format event as python objects, and provides tools
for packing and unpacking from raw bytes.

Note that using a numpy array for bank contents (rather than a regular python list/tuple)
will result in much faster encoding of the event when sending to midas buffers etc.
"""

import struct
import midas
import ctypes
import time

try:
    import numpy as np
    have_numpy = True
except ImportError:
    have_numpy = False
    
event_header_size = 16
all_bank_header_size = 8

class EventHeader:
    """
    Represents a midas EVENT_HEADER struct.
    See https://midas.triumf.ca/MidasWiki/index.php/Event_Structure#MIDAS_Format_Event
    
    Members:
        
    * event_id (int)
    * trigger_mask (int)
    * serial_number (int)
    * timestamp (int) - UNIX timestamp of event
    * event_data_size_bytes (int) - Size of all banks
    """
    def __init__(self):
        self.event_id = None
        self.trigger_mask = 0
        self.serial_number = None
        self.timestamp = None
        self.event_data_size_bytes = None     
        
    def dump(self):
        """
        Print the content of this event header to screen.
        """
        size_str = "%d/0x%x" % (self.event_data_size_bytes, self.event_data_size_bytes)
        print("Evid:%04d- Mask:%04d- Serial:%d- Time:0x%o- Dsize:%s" % (self.event_id, self.trigger_mask, self.serial_number, self.timestamp, size_str))
        

    def is_midas_internal_event(self):
        """
        Whether this is a special event that contains the ODB dumps or midas messages.
        
        Returns:
            bool
        """
        return self.is_bor_event() or self.is_eor_event() or self.is_msg_event()
        
    def is_bor_event(self):
        """
        Whether this is a special event that contains the begin-of-run ODB dump.
        
        Returns:
            bool
        """
        return self.event_id == 0x8000
    
    def is_eor_event(self):
        """
        Whether this is a special event that contains the end-of-run ODB dump.
        
        Returns:
            bool
        """
        return self.event_id == 0x8001
    
    def is_msg_event(self):
        """
        Whether this is a special event that contains a midas message.
        
        Returns:
            bool
        """
        return self.event_id == 0x8002
    
    def fill_from_struct_pointer(self, event_header_p):
        """
        Fill this `EventHeader` object from a C struct.
        This is a low-level interface that may be needed to handle some of
        the midas C library interfaces.
        
        Args:
            * event_header_p (`ctypes.POINTER(midas.structs.EventHeader)`)
            
        Returns:
            None (but the attributes of this object have been filled)
        """
        # Unpack the header, which was provided as a pointer to a C-struct
        raw_p = ctypes.cast(event_header_p, ctypes.POINTER(midas.structs.EventHeader))
        self.header.event_id = raw_p.contents.event_id
        self.header.trigger_mask = raw_p.contents.trigger_mask
        self.header.serial_number = raw_p.contents.serial_number
        self.header.timestamp = raw_p.contents.timestamp
        self.header.event_data_size_bytes = raw_p.contents.event_data_size_bytes
        
    def fill_from_bytes(self, header_data):
        """
        Fill this `EventHeader` object from a set of bytes.
        
        Args:
            * header_data (byte array of length 16)
            
        Returns:
            None (but the attributes of this object have been filled)
        """
        unpacked = struct.unpack(midas.endian_format_flag + "HHIII", header_data)
        self.event_id = unpacked[0]
        self.trigger_mask = unpacked[1]
        self.serial_number = unpacked[2]
        self.timestamp = unpacked[3]
        self.event_data_size_bytes = unpacked[4]

    def pack(self, buf=None, buf_offset=0):
        """
        Pack this header data into a buffer of bytes (creating the buffer if needed).
        If self.timestamp has not been set yet, we'll set it the current time. All 
        other attributes must be set.
        
        Args:
            * buf - The buffer to write to. Probably best created using
                `ctypes.create_string_buffer()`. If None, we'll create a buffer
                of the appropriate size.
            * buf_offset (int) - Where in the buffer to start writing this
                event.
                
        Returns:
            The buffer that was written to
        """
        if self.timestamp is None:
            self.timestamp = int(time.time())
            
        if self.event_id is None:
            raise ValueError("Must set event ID!")
        
        if self.trigger_mask is None:
            raise ValueError("Must set trigger mask!")
        
        if self.serial_number is None:
            raise ValueError("Must set serial number!")
        
        if self.event_data_size_bytes is None:
            raise ValueError("Must set event data size!")
        
        if buf is None:
            buf = ctypes.create_string_buffer(event_header_size)
            buf_offset = 0
        
        fmt = midas.endian_format_flag + "HHIII"
        struct.pack_into(fmt, buf, buf_offset, self.event_id, self.trigger_mask, self.serial_number, self.timestamp, self.event_data_size_bytes)

        return buf
    
class Bank:
    """
    Represents a midas BANK or BANK32 struct.
    See https://midas.triumf.ca/MidasWiki/index.php/Event_Structure#MIDAS_Format_Event
    
    Members:
        
    * name (str) - 4 characters
    * type (int) - See `TID_xxx` members in `midas` module
    * size_bytes (int)
    * data (tuple of int/float/byte etc, or a numpy array if use_numpy is specified when unpacking)
    """
    def __init__(self):
        self.name = None
        self.type = None
        self.size_bytes = None
        self.data = None

    def dump(self, data_idx_start=None, data_idx_end=None):
        """
        Dump the content of this bank to screen. Format closely matches that of
        the `mdump` utility.
        
        Some banks contain thousands of elements / entries. If you only want to
        see a few entries, you can specify a range to display. By default we
        show all of them.
        
        Args:
            * data_idx_start (int) - See above
            * data_idx_end (int) - See above
        """
        print("Bank:%s Length: %d bytes/%d entries Type:%s" % (self.name, self.size_bytes, len(self.data), midas.tid_texts.get(self.type, "Unknown")))
        printable_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        
        py2 = False
        
        try:
            # Python 3
            printable_bytes = bytes(printable_chars, "ascii")
        except:
            # Python 2
            printable_bytes = bytes(printable_chars)
            py2 = True
        
        if data_idx_start is None:
            data_idx_start = 1
        if data_idx_end is None:
            data_idx_end = len(self.data) + 1

        for start in range(data_idx_start - 1, data_idx_end - 1, 8):
            s = "%4d-> " % (start + 1)
            for i in range(start, min(len(self.data), start + 8)):
                # Use the style of default mdump
                if self.type == midas.TID_DOUBLE:
                    s += "%15.5e    " % self.data[i]
                elif self.type == midas.TID_FLOAT:
                    s += "%8.3e " % self.data[i]
                elif self.type == midas.TID_QWORD:
                    s += "0x%16.16x " % self.data[i]
                elif self.type == midas.TID_INT64:
                    s += "%16.1i " % self.data[i]
                elif self.type == midas.TID_DWORD:
                    s += "0x%8.8x " % self.data[i]
                elif self.type == midas.TID_INT:
                    s += "%8.1i " % self.data[i]
                elif self.type == midas.TID_WORD:
                    s += "0x%4.4x " % self.data[i]
                elif self.type == midas.TID_SHORT:
                    s += "%5.1i " % self.data[i]
                elif self.type == midas.TID_BYTE or self.type == midas.TID_STRUCT:
                    s += "0x%2.2x " % self.data[i]
                elif self.type == midas.TID_SBYTE:
                    s += "%4.i " % self.data[i]
                elif self.type == midas.TID_BOOL:
                    s += "Y " if self.data[i] else "N "
                elif self.type == midas.TID_CHAR:
                    if py2:
                        printable = chr(self.data[i]) in printable_bytes
                    else:
                        printable = self.data[i] in printable_bytes
                        
                    if printable:
                        s += str(chr(self.data[i])) + "    "
                    else:
                        s += "\\x%02x " % self.data[i]
                else:
                    s += "%s " % self.data[i]
                    
            print(s)
   
    def get_expected_padding(self):
        """
        Midas aligns banks to the nearest 8 bytes. This function tells you how
        much extra padding to put at the end of this bank's content.
        """
        return ((self.size_bytes + 7) & ~7) - self.size_bytes
    
    def get_pack_fmt(self, is_bank_32, is_64bit_aligned):
        """
        The format string to pass to struct.unpack to extract the bank header
        data.
        
        Args:
            * is_bank_32 (bool) - Whether the bank size is given by a 32-bit 
                number of a 16-bit number.
            * is_64bit_aligned (bool) - Whether bank data starts at 64-bit
                alignment
        """
        fmt = midas.endian_format_flag + "cccc"
        
        if is_bank_32:
            fmt += "II"
            
            if is_64bit_aligned:
                fmt += "I"
        else:
            fmt += "HH"
                
        return fmt

    def fill_header_from_bytes(self, bank_header_data, is_bank_32, is_64bit_aligned):
        """
        Fill this `EventHeader` object from a set of bytes.
        
        Args:
            * bank_header_data (byte array of length 8/12) - 8 bytes if 
                is_bank_32 is False; 12 bytes if is_bank_32 is True.
            * is_bank_32 (bool) - Whether the bank size is given by a 32-bit 
                number of a 16-bit number.
            * is_64bit_aligned (bool) - Whether bank data starts at 64-bit
                alignment
            
        Returns:
            None (but the attributes of this object have been filled)
        """
        unpacked = struct.unpack(self.get_pack_fmt(is_bank_32, is_64bit_aligned), bank_header_data)
        
        self.name = "".join(x.decode('ascii') for x in unpacked[:4])
        self.type = unpacked[4]
        self.size_bytes = unpacked[5]
        
        if self.type not in midas.tid_sizes:
            raise ValueError("Unexpected bank type %d for name '%s'" % (self.type, self.name))
        
    def convert_and_store_data(self, raw_data, use_numpy=False):
        """
        Fill self.data, converting the raw bytes to a list of appropriate 
        python data types.
        
        Args:
            * raw_data (byte array)
            * use_numpy (bool) - Whether to use numpy for extraction
        """
        if midas.tid_sizes[self.type] == 0:
            # No special handling - just return raw bytes.
            self.data = raw_data
        else:
            num_vals = int(self.size_bytes / midas.tid_sizes[self.type])
            
            if use_numpy:
                # Use numpy
                if midas.tid_np_formats[self.type] is None:
                    # No special handling - just return raw bytes.
                    self.data = raw_data
                else:
                    dt = midas.get_np_dtype(self.type)
                    
                    self.data = np.frombuffer(raw_data, dt, num_vals)
                        
                    if self.type == midas.TID_BOOL:
                        # Convert from 0/1 to False/True
                        self.data = self.data.astype(np.bool_)
            else:
                # Use tuples
                if midas.tid_unpack_formats[self.type] is None:
                    # No special handling - just return raw bytes.
                    self.data = raw_data
                else:
                    fmt = "%s%i%s" % (midas.endian_format_flag, num_vals, midas.tid_unpack_formats[self.type])
                    self.data = struct.unpack(fmt, raw_data)
                        
                    if self.type == midas.TID_BOOL:
                        # Convert from 0/1 to False/True
                        self.data = tuple(u != 0 for u in self.data)

class Event:
    """
    Represents a full midas event.
    See https://midas.triumf.ca/MidasWiki/index.php/Event_Structure#MIDAS_Format_Event
    for documentation of the event structure.
    
    Members:
    * header (`EventHeader`) - Metadata about the event
    * all_bank_size_bytes (int)
    * flags (int)
    * banks (dict of {str: `Bank`}) - Keyed by bank name
    * non_bank_data (bytes or None) - Content of some special events that don't 
        use banks (e.g. begin-of-run ODB dump)
    """
    def __init__(self, bank32=True, align64=False):
        """
        Args:
            * bank32 (bool) - 32-bit banks
            * align64 (bool) - Make bank data start at 64-bit alignment
        """
        self.header = EventHeader()
        self.all_bank_size_bytes = None
        self.flags = 1 # Data format v1
        
        self._flag_32bit = (1<<4)
        self._flag_align64 = (1<<5)
        
        if bank32:
            self.flags |= self._flag_32bit
            
        if align64:
            if not bank32:
                raise ValueError("16-bit banks with 64-bit alignment are not supported")
            
            self.flags |= self._flag_align64
        
        self.banks = {}
        self.non_bank_data = None
        
    def dump(self, include_bank_content=True):
        """
        Print the content of this event to screen. The output format closely
        resembles that of the `mdump` utility.
        
        Args:
            * include_bank_content (bool) - Whether to print the content of the
                data banks, or just the metadata.
        """
        print("-" * 30)
        self.header.dump()
        print("#banks:%d - Bank list:-%s-" % (len(self.banks), "".join(self.banks.keys())))
        
        if include_bank_content:
            for bank in self.banks.values():
                print()
                bank.dump()
                        
    def add_bank(self, bank):
        """
        Add a bank that was manually created to this event.
        Most people will use `create_bank()` instead of `add_bank()`.
        
        Args:
            * bank (`Bank`)
        """
        self.banks[bank.name] = bank
        
    def create_bank(self, bank_name, data_type, data):
        """
        Take python data, create a bank from it, and add it to the this event.
        
        * For most data types you should pass in a list/tuple/numpy.ndarray of 
            numbers.
        * For TID_BYTE and TID_CHAR you should pass in bytes (created with bytes() 
            or bytearray()).
        * For TID_BOOL you can pass in a list/tuple of bools (True/False) or a 
            numpy.ndarray of bools.

        If performance is a strong factor (and you have large bank sizes), 
        you should use a numpy array instead of raw python lists. The correct 
        (most efficient) numpy data type for each midas TID_xxx type is shown in 
        the `midas.tid_np_formats` dict. E.g. 

        * `midas.tid_np_formats[midas.TID_DWORD]` is `numpy.uint32`
        * `midas.tid_np_formats[midas.TID_DOUBLE]` is `numpy.float64` etc.

        Using a numpy array of the correct data type will result in significantly 
        faster encoding when sending this event to midas buffers.
        
        Args:
            * bank_name (str) - Name of this bank (must be 4 characters long).
            * data_type (int) - See midas.TID_xxx (e.g. midas.TID_INT for 
                storing integers).
            * data (list/tuple/numpy.ndarray/bytearray) - The data for this bank. 
                See above for more details.
        """
        if not isinstance(bank_name, str):
            raise TypeError("Bank name should be a 4-char string")
        if len(bank_name) != 4:
            raise ValueError("Bank name should be a 4-char string")
        if data_type not in midas.tid_sizes:
            raise ValueError("Unknown bank type")
        
        allowed_types = [list, tuple]
        
        if have_numpy:
            allowed_types.append(np.ndarray)
        
        if data_type in [midas.TID_BYTE, midas.TID_CHAR]:
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("Data must be a bytes() or bytearray() for TID_BYTE/TID_CHAR")
        elif midas.tid_sizes[data_type] is None:
            raise ValueError("Unsupported bank type")
        elif not isinstance(data, tuple(allowed_types)):
            raise TypeError("Data must be a list/tuple/numpy array for this data type")
        
        bank = Bank()
        bank.name = bank_name
        bank.type = data_type
        bank.data = data
        self.add_bank(bank)
        
    def get_bank(self, bank_name):
        """
        Return the data bank in this event with the given name.
        
        Args:
            * bank_name (str)
            
        Returns:
            `Bank`, or None if not found.
        """
        return self.banks.get(bank_name, None)
    
    def bank_exists(self, bank_name):
        """
        Whether this event contains a bank of the given name.
        
        Args:
            * bank_name (str)
            
        Returns:
            bool
        """
        return bank_name in self.banks

    def fill_header_from_bytes(self, bank_header_data):
        """
        Fill the attributes of this `EventBody` object (excluding the bank
        data itself) from a set of bytes.
        
        Args:
            * bank_header_data (byte array of length 8)
            
        Returns:
            None (but the attributes of this object have been filled)
        """
        unpacked = struct.unpack(midas.endian_format_flag + "II", bank_header_data)
        self.all_bank_size_bytes = unpacked[0]
        self.flags = unpacked[1]
        
    def is_bank_data_64bit_aligned(self):
        """
        Whether bank data payload is 64-bit aligned or not.
        """
        return (self.flags & self._flag_align64) != 0
        
    def is_bank_32(self):
        """
        Whether the size of banks are stored as 16-bit or 32-bit integers.
        """
        return (self.flags & self._flag_32bit) != 0
    
    def get_bank_header_size(self):
        """
        Get the number of bytes needed to store the header of a `Bank` object.
        """
        if self.is_bank_data_64bit_aligned():
            return 16
        elif self.is_bank_32():
            return 12
        else:
            return 8
        
    def calculate_bank_sizes(self):
        """
        Calculate and fill the size of the banks in this event, and the overall
        event size.
        
        This function must be called before we send events into a buffer.
        """
        self.all_bank_size_bytes = 0
        
        for bank in self.banks.values():
            # Bank size is simple
            bank.size_bytes = len(bank.data) * midas.tid_sizes[bank.type]
            
            # Total size adds padding and header size
            self.all_bank_size_bytes += bank.size_bytes + bank.get_expected_padding() + self.get_bank_header_size()

    def populate_bank_and_event_size(self):
        if self.all_bank_size_bytes is None:
            self.calculate_bank_sizes()
            
        # Note - this does not include the overall event header!
        if self.header.event_data_size_bytes is None:
            self.header.event_data_size_bytes = self.all_bank_size_bytes + all_bank_header_size
        
    def pack(self, buf=None, buf_offset=0):
        """
        Pack an event into a buffer of bytes (creating the buffer if needed).
        
        Args:
            * buf - The buffer to write to. Probably best created using
                `ctypes.create_string_buffer()`. If None, we'll create a buffer
                of the appropriate size.
            * buf_offset (int) - Where in the buffer to start writing this
                event.
                
        Returns:
            The buffer that was written to
        """
        self.populate_bank_and_event_size()

        if buf is None:
            buf_size = self.header.event_data_size_bytes + event_header_size
            buf = ctypes.create_string_buffer(buf_size)
            buf_offset = 0

        self.header.pack(buf, buf_offset)
        buf_offset += event_header_size
        
        # Overall header
        struct.pack_into(midas.endian_format_flag + "II", buf, buf_offset, self.all_bank_size_bytes, self.flags)
        buf_offset += all_bank_header_size
        
        for bank in self.banks.values():
            fmt = bank.get_pack_fmt(self.is_bank_32(), self.is_bank_data_64bit_aligned())
            
            # Need to make each char a bytes object of len 1
            name_bytes = [bytes(bank.name[i].encode("ascii")) for i in range(4)]
            header_info = [name_bytes[0], 
                           name_bytes[1], 
                           name_bytes[2], 
                           name_bytes[3], 
                           bank.type,
                           bank.size_bytes]
            
            if self.is_bank_data_64bit_aligned():
                # Extra reserved word to get 64-bit alignment
                header_info.append(0)
                
            struct.pack_into(fmt, buf, buf_offset, *header_info)
            
            buf_offset += self.get_bank_header_size()
            
            fmt = "%s%i%s" % (midas.endian_format_flag, len(bank.data), midas.tid_unpack_formats[bank.type])
            
            packed = False
            
            if have_numpy:
                if isinstance(bank.data, np.ndarray):
                    if bank.data.dtype == np.bool_:
                        # Avoid a deprecation warning issued when accessing a numpy
                        # array of np.bool_ like *(bank.data).
                        struct.pack_into(fmt, buf, buf_offset, *(bool(x) for x in bank.data))
                        packed = True
                    elif len(bank.data) > 0 and isinstance(bank.data[0], midas.tid_np_formats[bank.type]):
                        # Write directly to buffer
                        np_buf = bank.data.tobytes()
                        ctypes.memmove(ctypes.addressof(buf) + buf_offset, np_buf, len(np_buf))
                        packed = True
                    elif len(bank.data) > 0 and isinstance(bank.data[0], midas.get_np_dtype(bank.type)):
                        # Flip endianness of this big-endian data, then write directly to buffer
                        new_dtype = midas.get_np_dtype(bank.type)
                        np_buf = bank.data.asarray(new_dtype).tobytes()
                        ctypes.memmove(ctypes.addressof(buf) + buf_offset, np_buf, len(np_buf))
                        packed = True
                    
            if not packed:
                # Either not numpy, or numpy dtype doesn't match the requirements of this midas TID.
                # Do it the long way and pack into a struct.
                struct.pack_into(fmt, buf, buf_offset, *(bank.data))
            
            buf_offset += bank.size_bytes + bank.get_expected_padding()
            
        return buf
        
    def unpack(self, buf, buf_offset=0, use_numpy=False):
        """
        Unpack a buffer of bytes into this `Event` object.
        
        Args:
            * buf - The buffer to read from.
            * buf_offset - Location in the buffer where this event starts.
            * use_numpy (bool) - Whether to use numpy arrays or regular python tuples for bank data.
        
        Returns:
            None (but we've populated self.header and self.banks)
        """
        self.header.fill_from_bytes(buf[buf_offset:buf_offset+event_header_size])
        self.unpack_body(buf, buf_offset + event_header_size, use_numpy)
        
    def unpack_body(self, buf, buf_offset=0, use_numpy=False):
        """
        Unpack a buffer of bytes into this `Event` object. You must already have unpacked
        the event header into self.header.
        
        Args:
            * buf - The buffer to read from.
            * buf_offset - Location in the buffer where the overall bank header data starts.
            * use_numpy (bool) - Whether to use numpy arrays or regular python tuples for bank data.
        
        Returns:
            None (but we've populated self.banks)
        """
        if self.header is None or self.header.event_data_size_bytes is None:
            raise RuntimeError("Can't unpack event body without first unpacking header")
        
        orig_buf_offset = buf_offset
        
        if self.header.is_midas_internal_event():
            self.non_bank_data = buf[buf_offset:]
        else:
            all_bank_header_data = buf[buf_offset:buf_offset+midas.event.all_bank_header_size]
            buf_offset += midas.event.all_bank_header_size
            self.fill_header_from_bytes(all_bank_header_data)
            
            while (buf_offset - orig_buf_offset) < self.header.event_data_size_bytes - 4:
                bank_header_data = buf[buf_offset:buf_offset+self.get_bank_header_size()]
                buf_offset += self.get_bank_header_size()
                
                bank = midas.event.Bank()
                bank.fill_header_from_bytes(bank_header_data, self.is_bank_32(), self.is_bank_data_64bit_aligned())
                    
                raw_data = buf[buf_offset:buf_offset+bank.size_bytes]
                buf_offset += bank.size_bytes
                bank.convert_and_store_data(raw_data, use_numpy)
                
                self.add_bank(bank)

                buf_offset += bank.get_expected_padding()
                
