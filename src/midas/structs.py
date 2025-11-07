"""
Representations of important midas structs as ctypes.Structure objects.
"""

import ctypes

NAME_LENGTH = 32

class Key(ctypes.Structure):
    """
    The midas KEY struct.
    """
    _fields_ = [("type", ctypes.c_uint),
               ("num_values", ctypes.c_int),
               ("name", ctypes.c_char * NAME_LENGTH),
               ("data", ctypes.c_int),
               ("total_size", ctypes.c_int),
               ("item_size", ctypes.c_int),
               ("access_mode", ctypes.c_ushort),
               ("notify_count", ctypes.c_ushort),
               ("next_key", ctypes.c_int),
               ("parent_keylist", ctypes.c_int),
               ("last_written", ctypes.c_int)]
    
    def __str__(self):
        return "Key name %s of type %s" % (self.name.decode("utf-8"), self.type)
    
class EventHeader(ctypes.Structure):
    """
    The midas EVENT_HEADER struct.
    """
    _fields_ = [("event_id", ctypes.c_short),
                ("trigger_mask", ctypes.c_short),
                ("serial_number", ctypes.c_uint),
                ("time_stamp", ctypes.c_uint),
                ("data_size", ctypes.c_uint)]


