"""
This module contains a simple tool that helps you talk to LeCroy
oscilloscopes over a raw TCP connection, without using NI-VISA,
VISA passport, VXI-11 etc. It implements the LeCroy VICP protocol.

It is based on C code from http://optics.eee.nottingham.ac.uk/lecroy_tcp/,
but expanded slightly based on the latest VICP specification.

The code is very simplistic, and does not support all features (e.g. no
out-of-band communication, no serial poll, no sequence numbers).

Example usage would be:

```
instr = LeCroyInstrument("142.90.119.118")
print(instr.ask("*IDN?"))
print(instr.ask("CRS?"))
```
"""

import socket
import ctypes
import time

class LeCroyHeader(ctypes.Structure):
    """
    Structure that we must send before any of our queries, and that
    the scope responds with before sending its payload.
    """
    _fields_ = [("operation_flags", ctypes.c_uint8),
                ("header_version", ctypes.c_uint8),
                ("sequence_number", ctypes.c_uint8),
                ("reserved", ctypes.c_uint8),
                ("data_length", ctypes.c_int32)]
    
class LeCroyInstrument():
    def __init__(self, ip, port=1861):
        self.EOI_FLAG     = 0x01 # Whether to use EOI terminator or not
        self.SRQ_FLAG     = 0x08 # Only ever sent from device to us
        self.CLEAR_FLAG   = 0x10 # Clear device before parsing our data
        self.LOCKOUT_FLAG = 0x20 # Lock out the front panel
        self.REMOTE_FLAG  = 0x40 # Remote mode
        self.DATA_FLAG    = 0x80 # Data block
        
        self.header_version = 1
        
        self.connect_timeout_secs = 5
        self.read_timeout_secs = 3

        self.ip = ip
        self.port = port
        
        self.conn = socket.socket()
        self._connect()
    
    def _connect(self):
        # We connect automatically in the constructor
        self.conn.settimeout(self.connect_timeout_secs)
        self.conn.connect((self.ip, self.port))
        self.conn.settimeout(self.read_timeout_secs)

    def disconnect(self):
        self.conn.close()
        
    def clear(self):
        header = LeCroyHeader()
        header.operation_flags = self.CLEAR_FLAG
        header.header_version = self.header_version
        
        self.conn.send(header)
        time.sleep(0.1)
        self.disconnect()
        self._connect()

    def write(self, cmd_str):
        payload = bytes(cmd_str, "ascii")
        header = LeCroyHeader()
        header.operation_flags = self.DATA_FLAG | self.EOI_FLAG
        header.header_version = self.header_version
        header.data_length = socket.htonl(len(payload))
        
        self.conn.send(header)
        self.conn.send(payload)
        
    def read(self, stringify_response=True):
        header = LeCroyHeader()
        self.conn.recv_into(header)
        
        if header.data_length:
            retval = self.conn.recv(header.data_length)
        else:
            retval = b""
        
        if stringify_response:
            return retval.decode("ascii").strip()
        else:
            return retval
        
    def ask(self, cmd_str, stringify_response=True):
        self.write(cmd_str)
        return self.read(stringify_response)
    