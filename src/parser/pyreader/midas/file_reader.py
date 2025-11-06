"""
Tools to read a midas file.

The midas file format is documented at https://midas.triumf.ca/MidasWiki/index.php/Event_Structure#MIDAS_Format_Event.
Each event has a "header" containing metadata, then a series of names "banks" containing the actual data.
The banks can be of different formats (e.g. lists of floats/doubles/ints).

The tools in this files can read in midas events, and automatically convert the data in the banks
to appropriate python data types (so you get a tuple of ints, rather than just a raw set of bytes).

We can read on files in .mid. .mid.gz and .mid.lz4 format. lz4 support is not present in the standard python
libraries, but can be installed using pip (e.g. `pip install lz4`). See the README file to learn more about pip.


Basic usage:

```
import midas.file_reader

# Open our file
mfile = midas.file_reader.MidasFile("040644.mid")

# We can simply iterate over all events in the file
for event in mfile:
    bank_names = ", ".join(b.name for b in event.banks.values())
    print("Event # %s of type ID %s contains banks %s" % (event.header.serial_number, event.header.event_id, bank_names))
```


More complex usage, which can allow you to skip events more efficiently:

```
import midas.file_reader

# Open our file
mfile = midas.file_reader.MidasFile("040644.mid")

# Here we choose to just read in the header of each event, and will read
# the body (the actual banks) later.
while mfile.read_next_event_header():
    header = mfile.event.header
    
    if header.is_midas_internal_event():
        # Skip over events that contain midas messages or ODB dumps
        continue
    
    print("Overall size of event # %s of type ID %s is %d bytes" % (header.serial_number, header.event_id, header.event_data_size_bytes))
    
    if not mfile.read_this_event_body():
        raise RuntimeError("Unexpectedly failed to read body of event!")
    
    # Loop over the banks of data in this event and print information about them
    for name, bank in mfile.event.banks.items():
        # The `bank.data` member is automatically converted to appropriate python data types.
        # Here we're just figuring out what that type is to print it to screen. Normally
        # you already know what to expect for each bank, and could just the tuple of floats,
        # for example.
        
        if isinstance(bank.data, tuple) and len(bank.data):
            # A tuple of ints/floats/etc (a tuple is like a fixed-length list)
            type_str = "tuple of %s containing %d elements" % (type(bank.data[0]).__name__, len(bank.data))
        elif isinstance(bank.data, tuple):
            # A tuple of length zero
            type_str = "empty tuple"
        elif isinstance(bank.data, str):
            # Of the original data was a list of chars, we convert to a string.
            type_str = "string of length %d" % len(bank.data)
        else:
            # Some data types we just leave as a set of bytes.
            type_str = type(bank.data[0]).__name__
        
        print("  - bank %s contains %d bytes of data. Python data type: %s" % (name, bank.size_bytes, type_str))


```



Example usage for reading ODB information:

```
import midas.file_reader

mfile = midas.file_reader.MidasFile("040129.mid")

try:
    # Try to find the special midas event that contains an ODB dump.
    odb = mfile.get_bor_odb_dump()
    
    # The full ODB is stored as a nested dict withing the `odb.data` member.
    run_number = odb.data["Runinfo"]["Run number"]
    print("We are looking at a file from run number %s" % run_number)
except RuntimeError:
    # No ODB dump found (mlogger was probably configured to not dump
    # the ODB at the start of each subrun).
    print("No begin-of-run ODB dump found")

```

"""
import gzip
import bz2
import struct
import midas
import midas.event
import datetime
import math
from xml.etree import ElementTree

try:
    import lz4.frame
    have_lz4 = True
except ImportError:
    have_lz4 = False
        
class MidasFile:
    """
    Provides access to a midas file - either raw (.mid), gzipped (.mid.gz) or lz4 (.mid.lz4).
    
    Members:
        
    * file (file-like object)
    * event (`Event`) - The event we've just read
    * next_event_offset (int) - Position in file where the next event starts
    * this_event_payload_offset (int) - Sometimes we just read the event header,
        not the full data. This member is where the data of the current event starts.
    * use_numpy (bool) - Whether to use numpy when extracting bank contents (so bank
        data is a numpy array rather than a standard python tuple)
    """
    def __init__(self, path, use_numpy=False):
        """
        Open a midas file.
        
        Args:
            
        * path (str) - Path to the file
        * use_numpy (bool) - Whether to use numpy when extracting bank contents (so bank
            data is a numpy array rather than a standard python tuple)
        """
        self.file = None
        self.event = None
        self.next_event_offset = 0
        self.this_event_payload_offset = 0
        self.use_numpy = use_numpy
        self.reset_event()
        self.open(path)
        
    def __enter__(self):
        """
        Allow use as `with MidasFile("xxx") as f:`
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Allow use as `with MidasFile("xxx") as f:`
        """
        if self.file:
            self.file.close()
            self.file = None
        
    def __del__(self):
        """
        Clean up file handle when we go out of scope.
        """
        if self.file:
            self.file.close()
        
    def __next__(self):
        """
        Iterable interface for looping through events.
        """
        ev = self.read_next_event()
        
        if not ev:
            raise StopIteration()
        else:
            return ev
    
    next = __next__ # for Python 2        
        
    def __iter__(self):
        """
        Iterable interface for looping through events.
        """
        return self
        
    def reset_event(self):
        """
        Forget about an event we've already read (but don't rewind
        the actual file pointer).
        """
        self.event = midas.event.Event()
        self.this_event_payload_offset = 0
        
    def open(self, path):
        """
        Open a midas file.
        
        Args:
            
        * path (str) - Path to midas file. Can be raw, gz or lz4 compressed.
        """
        self.reset_event()
        
        if path.endswith(".lz4"):
            if have_lz4:
                self.file = lz4.frame.LZ4FrameFile(path, "rb")
            else:
                raise ImportError("lz4 package not found - install using 'pip install lz4'")
        elif path.endswith(".gz"):
            self.file = gzip.open(path, "rb")
        elif path.endswith(".bz2"):
            self.file = bz2.open(path, "rb")
        else:
            self.file = open(path, "rb")
    
    def jump_to_start(self):
        """
        Rewind to the start of the file.
        """
        self.file.seek(0,0)
        self.next_event_offset = 0
        self.reset_event()
    
    def get_bor_odb_dump(self):
        """
        Return the begin-of-run ODB dump as a `midas.file_reader.Odb` object.
        
        Raises a RuntimeError if the dump can't be found.
        """
        self.jump_to_start()
        
        if self.read_next_event_header() and self.event.header.is_bor_event():
            self.read_this_event_body()
            return Odb(self.event.non_bank_data)
        
        self.jump_to_start()
        raise RuntimeError("Unable to find BOR event")
    
    def get_eor_odb_dump(self):
        """
        Return the end-of-run ODB dump as a `midas.file_reader.Odb` object.
        
        Raises a RuntimeError if the dump can't be found.
        """
        started_at_start = (self.next_event_offset == 0)
        read_any = False

        while True:
            if not self.read_next_event_header():
                # Reached the end of the file
                break

            read_any = True
        
            if self.event.header.is_eor_event():
                self.read_this_event_body()
                return Odb(self.event.non_bank_data)
        
        self.jump_to_start()

        if not read_any and not started_at_start:
            # We started at the end of the file (after the
            # EOR dump). Try to find it again now that we've
            # jumped back to the start.
            return self.get_eor_odb_dump()

        # We started before the end of the file and still weren't
        # able to find the EOR dump - it really doesn't exist.
        raise RuntimeError("Unable to find EOR event")
    
    def get_next_event_with_bank(self, bank_name):
        """
        Find the next event that contain a bank with the specified name.
        
        Returns:
            `Event`, or None of no such event found.
        """
        while self.read_next_event():
            if bank_name in self.event.banks.keys():
                return self.event
            
        return None
    
    def read_next_event(self):
        """
        Read the header and content of the next event.
        May be slow if there is a lot of data.
        
        Returns:
            `Event`, of None if no more events left.
        """
        if self.read_next_event_header():
            return self.read_this_event_body()
        else:
            return None
    
    def read_next_event_header(self):
        """
        Just read the header/metadata of the next event.
        If you read it and think it's interesting, you can then call
        read_this_event_body() to grab the actual data.
        If the event isn't interesting, then you saved yourself a lot
        of time by not loading a bunch of data you don't care about.
        
        Returns:
            `Event` (with only the header populated), or None if no
            more events left.
        """
        self.reset_event()
        this_event_offset = self.next_event_offset
        
        self.file.seek(self.next_event_offset, 0)
        header_data = self.file.read(midas.event.event_header_size)
        
        if not header_data:
            return None
        
        self.event.header.fill_from_bytes(header_data)
        
        self.this_event_payload_offset = this_event_offset + midas.event.event_header_size
        self.next_event_offset += self.event.header.event_data_size_bytes + midas.event.event_header_size
        
        return self.event
    
    def read_this_event_body(self):
        """
        Read the data of the current event (that you've already read the header info of).
        
        Populates event.banks or event.non_bank_data (depending on the event type).
        
        Returns:
            `Event` (with both the header and body populated)
        """
        self.file.seek(self.this_event_payload_offset, 0)
        body_data = self.file.read(self.event.header.event_data_size_bytes)
        self.event.unpack_body(body_data, 0, self.use_numpy)
        return self.event

    def get_event_count(self, include_midas_special_events=False):
        """
        Count the number of events in this file.
        
        Args:
            * include_midas_special_events (bool) - Whether to include
                midas' internal events in the count (begin-of-run, message and 
                end-of-run events).
        
        Returns:
            int
        """
        self.jump_to_start()
        count = 0
        
        while self.read_next_event_header():
            if self.event.header.is_midas_internal_event() and not include_midas_special_events:
                continue
            
            count += 1
            
        self.jump_to_start()
        return count

class Odb:
    """
    Helps read an XML/JSON representation of an ODB, and convert it to a python dict.
    
    Members:
        
    * written_time (datetime.datetime) - Time the ODB dump was written (only if the dump was written as XML)
    * data (dict) - The actual ODB structure
    """
    def __init__(self, odb_string = None):
        """
        Initialize an ODB object py parsing an ODB dump.
        
        Args:
            odb_string (str) - Either XML or JSON representation of an ODB dump. 
        """
        self.written_time = None 
        self.data = {}
        
        if odb_string is not None and len(odb_string) > 0:
            if odb_string[0] in ["<", 60]:
                # This decode/encode is needed so that we can handle non-ascii values
                # that may be in the dump.
                # The -1 is needed so the XML parser doesn't complain about an invalid token.
                self.load_from_xml_string(odb_string.decode('utf-8').encode('utf-8')[:-1])
            elif odb_string[0] in ["{", 123]:
                self.load_from_json_string(odb_string)
            else:
                raise ValueError("Couldn't determine ODB dump format (first character is '%s', rather than expected '<' or '{')" % odb_string[0])
        
    def load_from_json_string(self, json_string):
        """
        """
        self.written_time = None
        self.data = midas.safe_to_json(json_string)
        
    def load_from_xml_string(self, xml_string):
        """
        Parse the XML string to populate self.data and self.written_time.
        
        Args:
            
        * xml_string (bytes)
        """
        self.written_time = None
        self.data = {}
        
        """
        Header looks like:
        
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <!-- created by MXML on Fri Jan  4 10:51:22 2019 -->
        
        Extract the creation time.
        """
        comment_start = xml_string.find(b"<!--")
        
        if comment_start != -1:
            ts_start = xml_string.find(b" on ", comment_start)
            ts_end = xml_string.find(b" -->", comment_start)
            
            if ts_start != -1 and ts_end != -1:
                ts_str = xml_string[ts_start+4:ts_end].decode('utf-8')
                self.written_time = datetime.datetime.strptime(ts_str, "%c")
        
        """
        Now parse the actual XML.
        """
        root = ElementTree.fromstring(xml_string)
        self.handle_node(root, self.data)
       
    def text_to_value(self, text, type_str):   
        if type_str in ["INT", "INT8", "INT16", "INT32", "INT64"]:
            return int(text)
        elif type_str in ["WORD", "DWORD", "UINT16", "UINT32", "QWORD", "UINT64"]:
            return "0x%x" % int(text)
        elif type_str == "BOOL":
            return text == "y"
        elif type_str in ["STRING", "LINK"]:
            return text
        elif type_str in ["DOUBLE", "FLOAT"]:
            val = float(text)
            if math.isnan(val):
                return "NaN"
            return val
        else:
            raise ValueError("Unhandled ODB type %s" % type_str)
        
    def type_to_int(self, type_str):
        """
        Convert e.g. "INT" to "7", the midas code for TID_INT.
        
        Args:
            
        * type_str (str) INT/WORD/FLOAT etc
        
        Returns:
            int
        """
        try:
            return getattr(midas, "TID_" + type_str)
        except:
            raise ValueError("Unknown ODB type TID_%s" % type_str)
        
    def create_key_entry(self, node):
        """
        Metadata for node "X" is stored in an extra dict "X/key".
        
        Args:
            
        * node (`xml.etree.ElementTree.Element`)
        
        Returns:
            dict
        """
        type_str = node.attrib["type"]
        type_int = self.type_to_int(type_str)
        
        key_dict = {"type": type_int}
        
        if node.tag == "keyarray":
            key_dict["num_values"] = int(node.attrib["num_values"])
            
        if type_int == midas.TID_LINK:
            key_dict["link"] = node.text
            
        if type_int == midas.TID_STRING:
            key_dict["item_size"] = node.attrib["size"]
            
        return key_dict
            
    def handle_node(self, node, obj):
        """
        Called recursively to work through the whole XML tree,
        converting nodes to a nested dict.
        
        Args:
            
        * node (`xml.etree.ElementTree.Element`) - Current position in XML tree
        * obj (dict) - Object to add more elements to
        """
        for child in node:
            
            if child.tag == "dir":
                name = child.attrib["name"]
                obj[name] = {}
                self.handle_node(child, obj[name])
            elif child.tag == "keyarray":
                name = child.attrib["name"]
                type_str = child.attrib["type"]
                obj[name] = []
                obj[name + "/key"] = self.create_key_entry(child)
                
                for val_node in child:
                    val = self.text_to_value(val_node.text, type_str)
                    obj[name].append(val)
            elif child.tag == "key":
                name = child.attrib["name"]
                type_str = child.attrib["type"]
                val = self.text_to_value(child.text, type_str)
                obj[name] = val                
                obj[name + "/key"] = self.create_key_entry(child)
                
            else:
                raise ValueError("Unhandled tag %s" % child.tag)
                
