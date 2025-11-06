"""
This module contains python tools related to midas.

This file contains definitions that come from midas.h
"""

import sys
import json
import ctypes
import midas.structs
import collections
import logging

try:
    import numpy as np
    have_numpy = True
except ImportError:
    have_numpy = False

# Setup a logger that does nothing. Users (or the frontend class)
# can add their own handlers if desired.
logger = logging.getLogger('midas')
logger.addHandler(logging.NullHandler())

# Run states
STATE_STOPPED = 1
STATE_PAUSED  = 2
STATE_RUNNING = 3

# Message types
MT_ERROR = (1<<0)
MT_INFO  = (1<<1)
MT_DEBUG = (1<<2)
MT_USER  = (1<<3)
MT_LOG   = (1<<4)
MT_TALK  = (1<<5)
MT_CALL  = (1<<6)
MT_ALL   = 0xFF

# Data types                                   min      max    
TID_BYTE     =  1       # unsigned byte         0       255
TID_UINT8    =  TID_BYTE    
TID_SBYTE    =  2       # signed byte         -128      127    
TID_INT8     =  TID_SBYTE    
TID_CHAR     =  3       # single character      0       255    
TID_WORD     =  4       # two bytes             0      65535   
TID_UINT16   =  TID_WORD    
TID_SHORT    =  5       # signed word        -32768    32767   
TID_INT16    =  TID_SHORT
TID_DWORD    =  6       # four bytes            0      2^32-1  
TID_UINT32   =  TID_DWORD
TID_INT      =  7       # signed dword        -2^31    2^31-1  
TID_INT32    =  TID_INT
TID_BOOL     =  8       # four bytes bool       0        1     
TID_FLOAT    =  9       # 4 Byte float format                  
TID_FLOAT32  = TID_FLOAT
TID_DOUBLE   = 10       # 8 Byte float format                  
TID_FLOAT64  = TID_DOUBLE
TID_BITFIELD = 11       # 32 Bits Bitfield      0     111... (32) 
TID_STRING   = 12       # null-terminated string               
TID_ARRAY    = 13       # array with unknown contents          
TID_STRUCT   = 14       # structure with fixed length          
TID_KEY      = 15       # key in online database               
TID_LINK     = 16       # link in online database    
TID_INT64    = 17       # 8 bytes int          -2^63   2^63-1  */
TID_QWORD    = 18       # 8 bytes unsigned int  0      2^64-1  */
TID_UINT64   = TID_QWORD

# Alarm types
AT_INTERNAL  = 1
AT_PROGRAM   = 2
AT_EVALUATED = 3
AT_PERIODIC  = 4

# Run transitions
TR_START  = 1
TR_STOP   = 2
TR_PAUSE  = 4
TR_RESUME = 8
TR_STARTABORT = 16

TR_SYNC    = 1
TR_ASYNC   = 2
TR_DETACH  = 4
TR_MTHREAD = 8

# Buffer access asynchronous/synchronous modes
BM_WAIT    = 0
BM_NO_WAIT = 1

# Buffer sampling type
GET_ALL         = 1 # get all events (consume)
GET_NONBLOCKING = 2 # get as much as possible without blocking producer 
GET_RECENT      = 4 # get recent event (not older than 1 s)

# When frontends should send events (RO means "readout")
RO_RUNNING     = (1<<0)   # While running 
RO_STOPPED     = (1<<1)   # Before stopping the run 
RO_PAUSED      = (1<<2)   # While run is paused 
RO_BOR         = (1<<3)   # At the Begin of run 
RO_EOR         = (1<<4)   # At the End of run 
RO_PAUSE       = (1<<5)   # Before pausing the run 
RO_RESUME      = (1<<6)   # Before resuming the run 

RO_TRANSITIONS = (RO_BOR|RO_EOR|RO_PAUSE|RO_RESUME)      # At all transitions 
RO_ALWAYS      = (0xFF)   # Always (independent of the run status) 

RO_ODB         = (1<<8)   # Data should be sent to ODB as well as buffer

# Frontend equipment types
EQ_PERIODIC    = (1<<0)   # Periodic Event 
EQ_POLLED      = (1<<1)   # Polling Event 
EQ_INTERRUPT   = (1<<2)   # Interrupt Event 
EQ_MULTITHREAD = (1<<3)   # Multithread Event readout 
EQ_SLOW        = (1<<4)   # Slow Control Event 
EQ_MANUAL_TRIG = (1<<5)   # Manual triggered Event 
EQ_FRAGMENTED  = (1<<6)   # Fragmented Event 
EQ_EB          = (1<<7)   # Event run through the event builder 
EQ_USER        = (1<<8)   # Polling handled in user part 

MAX_STRING_LENGTH = 256 # Max ODB string length 

RPC_JRPC = 18000 # RPC code for JRPC (string)
RPC_BRPC = 18001 # RPC code for BRPC (binary data)

# Status codes used by midas
status_codes = {
    "SUCCESS": 1,
    "CM_SET_ERROR": 102,
    "CM_NO_CLIENT": 103,
    "CM_DB_ERROR": 104,
    "CM_UNDEF_EXP": 105,
    "CM_VERSION_MISMATCH": 106,
    "CM_SHUTDOWN": 107,
    "CM_WRONG_PASSWORD": 108,
    "CM_UNDEF_ENVIRON": 109,
    "CM_DEFERRED_TRANSITION": 110,
    "CM_TRANSITION_IN_PROGRESS": 111,
    "CM_TIMEOUT": 112,
    "CM_INVALID_TRANSITION": 113,
    "CM_TOO_MANY_REQUESTS": 114,
    "CM_TRUNCATED": 115,
    "CM_TRANSITION_CANCELED": 116,
    "BM_CREATED": 202,
    "BM_NO_MEMORY": 203,
    "BM_INVALID_NAME": 204,
    "BM_INVALID_HANDLE": 205,
    "BM_NO_SLOT": 206,
    "BM_NO_SEMAPHORE": 207,
    "BM_NOT_FOUND": 208,
    "BM_ASYNC_RETURN": 209,
    "BM_TRUNCATED": 210,
    "BM_MULTIPLE_HOSTS": 211,
    "BM_MEMSIZE_MISMATCH": 212,
    "BM_CONFLICT": 213,
    "BM_EXIT": 214,
    "BM_INVALID_PARAM": 215,
    "BM_MORE_EVENTS": 216,
    "BM_INVALID_MIXING": 217,
    "BM_NO_SHM": 218,
    "BM_CORRUPTED": 219,
    "DB_CREATED": 302,
    "DB_NO_MEMORY": 303,
    "DB_INVALID_NAME": 304,
    "DB_INVALID_HANDLE": 305,
    "DB_NO_SLOT": 306,
    "DB_NO_SEMAPHORE": 307,
    "DB_MEMSIZE_MISMATCH": 308,
    "DB_INVALID_PARAM": 309,
    "DB_FULL": 310,
    "DB_KEY_EXIST": 311,
    "DB_NO_KEY": 312,
    "DB_KEY_CREATED": 313,
    "DB_TRUNCATED": 314,
    "DB_TYPE_MISMATCH": 315,
    "DB_NO_MORE_SUBKEYS": 316,
    "DB_FILE_ERROR": 317,
    "DB_NO_ACCESS": 318,
    "DB_STRUCT_SIZE_MISMATCH": 319,
    "DB_OPEN_RECORD": 320,
    "DB_OUT_OF_RANGE": 321,
    "DB_INVALID_LINK": 322,
    "DB_CORRUPTED": 323,
    "DB_STRUCT_MISMATCH": 324,
    "DB_TIMEOUT": 325,
    "DB_VERSION_MISMATCH": 326,
    "SS_CREATED": 402,
    "SS_NO_MEMORY": 403,
    "SS_INVALID_NAME": 404,
    "SS_INVALID_HANDLE": 405,
    "SS_INVALID_ADDRESS": 406,
    "SS_FILE_ERROR": 407,
    "SS_NO_SEMAPHORE": 408,
    "SS_NO_PROCESS": 409,
    "SS_NO_THREAD": 410,
    "SS_SOCKET_ERROR": 411,
    "SS_TIMEOUT": 412,
    "SS_SERVER_RECV": 413,
    "SS_CLIENT_RECV": 414,
    "SS_ABORT": 415,
    "SS_EXIT": 416,
    "SS_NO_TAPE": 417,
    "SS_DEV_BUSY": 418,
    "SS_IO_ERROR": 419,
    "SS_TAPE_ERROR": 420,
    "SS_NO_DRIVER": 421,
    "SS_END_OF_TAPE": 422,
    "SS_END_OF_FILE": 423,
    "SS_FILE_EXISTS": 424,
    "SS_NO_SPACE": 425,
    "SS_INVALID_FORMAT": 426,
    "SS_NO_ROOT": 427,
    "SS_SIZE_MISMATCH": 428,
    "SS_NO_MUTEX": 429,
    "RPC_NO_CONNECTION": 502,
    "RPC_NET_ERROR": 503,
    "RPC_TIMEOUT": 504,
    "RPC_EXCEED_BUFFER": 505,
    "RPC_NOT_REGISTERED": 506,
    "RPC_CONNCLOSED": 507,
    "RPC_INVALID_ID": 508,
    "RPC_SHUTDOWN": 509,
    "RPC_NO_MEMORY": 510,
    "RPC_DOUBLE_DEFINED": 511,
    "RPC_MUTEX_TIMEOUT": 512,
    "FE_ERR_ODB": 602,
    "FE_ERR_HW": 603,
    "FE_ERR_DISABLED": 604,
    "FE_ERR_DRIVER": 605,
    "FE_PARTIALLY_DISABLED": 606,
    "HS_FILE_ERROR": 702,
    "HS_NO_MEMORY": 703,
    "HS_TRUNCATED": 704,
    "HS_WRONG_INDEX": 705,
    "HS_UNDEFINED_EVENT": 706,
    "HS_UNDEFINED_VAR": 707,
    "FTP_NET_ERROR": 802,
    "FTP_FILE_ERROR": 803,
    "FTP_RESPONSE_ERROR": 804,
    "FTP_INVALID_ARG": 805,
    "EL_FILE_ERROR": 902,
    "EL_NO_MESSAGE": 903,
    "EL_TRUNCATED": 904,
    "EL_FIRST_MSG": 905,
    "EL_LAST_MSG": 906,
    "AL_INVALID_NAME": 1002,
    "AL_ERROR_ODB": 1003,
    "AL_RESET": 1004,
    "AL_TRIGGERED": 1005
}

# Reverse mapping of status_codes - from int to string
status_codes_to_text = {v: k for k,v in status_codes.items()}


# Number of bytes each midas data type requires
tid_sizes = {  TID_BYTE: 1, 
               TID_SBYTE: 1,
               TID_CHAR: 1, 
               TID_WORD: 2,
               TID_SHORT: 2,
               TID_DWORD: 4,
               TID_INT: 4,
               TID_BOOL: 4,
               TID_FLOAT: 4,
               TID_DOUBLE: 8,
               TID_BITFIELD: 4,
               TID_STRING: None,
               TID_ARRAY: None,
               TID_STRUCT: None,
               TID_KEY: None,
               TID_LINK: None,
               TID_INT64: 8,
               TID_QWORD: 8
            }

# How to unpack each midas data type with python's struct module
tid_unpack_formats = {  TID_BYTE: 'B', # C char / python int
                        TID_SBYTE: 'b', # C signed char / python int
                        TID_CHAR: 'B', # C char / we'll make a python string 
                        TID_WORD: 'H', # C unsigned short / python int
                        TID_SHORT: 'h', # C signed short / python int
                        TID_DWORD: 'I', # C unsigned int / python int
                        TID_INT: 'i', # C signed int / python int
                        TID_BOOL: 'I', # C unsigned int / we'll make a list of python bools
                        TID_FLOAT: 'f', # C float / python float
                        TID_DOUBLE: 'd', # C double / python double
                        TID_BITFIELD: 'I', # C unsigned int / python int 
                        TID_STRING: None, # We just give raw bytes
                        TID_ARRAY: None, # We just give raw bytes
                        TID_STRUCT: None, # We just give raw bytes
                        TID_KEY: None, # We just give raw bytes
                        TID_LINK: None, # We just give raw bytes
                        TID_QWORD: 'Q', # C unsigned long long / python int
                        TID_INT64: 'q', # C signed long long / python int
                    }

if have_numpy:
    tid_np_formats = {  TID_BYTE: np.uint8,
                        TID_SBYTE: np.int8,
                        TID_CHAR: np.uint8,
                        TID_WORD: np.uint16,
                        TID_SHORT: np.int16,
                        TID_DWORD: np.uint32,
                        TID_INT: np.int32,
                        TID_BOOL: np.uint32, # We'll convert to np.bool_ later
                        TID_FLOAT: np.float32,
                        TID_DOUBLE: np.float64,
                        TID_BITFIELD: np.uint32,
                        TID_STRING: None, # We just give raw bytes
                        TID_ARRAY: None, # We just give raw bytes
                        TID_STRUCT: None, # We just give raw bytes
                        TID_KEY: None, # We just give raw bytes
                        TID_LINK: None, # We just give raw bytes
                        TID_QWORD: np.uint64,
                        TID_INT64: np.int64
                    }

# Friendly name of each midas data type
tid_texts = {  TID_BYTE: "Unsigned Byte", 
               TID_SBYTE: "Signed Byte",
               TID_CHAR: "Char", 
               TID_WORD: "Unsigned Word",
               TID_SHORT: "Signed Word",
               TID_DWORD: "Unsigned Integer",
               TID_INT: "Signed Integer",
               TID_BOOL: "Boolean",
               TID_FLOAT: "Float",
               TID_DOUBLE: "Double",
               TID_BITFIELD: "Bitfield",
               TID_STRING: "String",
               TID_ARRAY: "Array",
               TID_STRUCT: "Struct",
               TID_KEY: "Key",
               TID_LINK: "Link",
               TID_QWORD: "Unsigned 64-bit Integer",
               TID_INT64: "Signed 64-bit Integer"
            }

# Read in little-endian by default
endian_format_flag = "<"

def sys_is_big_endian():
    return sys.byteorder != 'little'

def get_np_dtype(midas_tid):
    if sys_is_big_endian():
        dt = np.dtype(tid_np_formats[midas_tid])
        dt.newbyteorder(endian_format_flag)
    else:
        dt = tid_np_formats[midas_tid]
    
    return dt

class MidasError(Exception):
    """
    Exception type raised if the midas C library functions return a 
    non-SUCCESS code.
    
    Members:
        * code (int) - The return code
        * message (str) - Meaning of the return code, as defined in `status_codes`.
    """
    def __init__(self, code, message):
        self.code = code
        self.message = message

class TransitionFailedError(MidasError):
    """
    A specific exception raised if this client tries to start/stop a run,
    but the transition fails.
    """
    pass

class TransitionDeferredError(MidasError):
    """
    A specific exception raised if this client tries to start/stop a run,
    but the transition is deferred by another client.
    """
    pass

class MidasShutdown(MidasError):
    """
    A specific exception raised when midas tells this client to shutdown.
    Either due to Ctrl-C in terminal or via RPC.
    """
    pass

class MidasLogHandler(logging.Handler):
    """
    This handler integrates with the python logging library, and will automatically 
    write any messages to the midas message log.

    Usage is like:

    ```
    import logging
    import midas.client

    client = midas.client.MidasClient("pytest")
    logger = logging.getLogger("mylogger") # Or anything else
    handler = midas.MidasLogHandler(client) # Also has an optional 2nd argument for the message "facility"

    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Ask to log info/error messages, but not debug ones

    logger.info("Some message that will be shown in stdout and in midas log")
    logger.debug("This will not be shown or logged, as debug < info")
    logger.error("This will be shown in stdout and in midas log (as an error message)")
    ```
    """
    def __init__(self, client=None, facility='midas'):
        super().__init__()
        self.client = client
        self.facility = facility

    def emit(self, record):
        if self.client is None:
            return

        msg = self.format(record)
        is_error = record.levelno >= logging.ERROR
        self.client.msg(msg, is_error, facility=self.facility)

class MidasLib(ctypes.CDLL):
    """
    Wrapper around the midas C library that will automatically raise
    exceptions if a called function returns a non-SUCCESS status code.
    
    Members:
        * return_types (dict of {str: <type>} for {function_name:
            return type}) - For functions that do NOT return a normal
            integer status code, what the return type is. Most users will
            not need to call any of these functions, but we record the types
            any way for completeness.
        * ignore_status_codes (dict of {str: [str]} for {function_name:
            list_of_statuses}) - Which return codes should NOT raise an
            exception for certain functions.
        * db_key_name_arg (dict of {str: int} for {function_name: 
            argument_index}) - For certain functions we raise a KeyError if
            a path is not found in the ODB. This mapping tells us which 
            functions to apply that logic to, and which argument contains
            the path we were searching for (so we can give a nicer error
            message).
    """
    def __init__(self, lib_path):
        """
        Args:
            * lib_path (str) - Path to the shared library to load.
        """
        logger.info("Opening midas shared library %s" % lib_path)
        ctypes.CDLL.__init__(self, lib_path)
        
        # This dict only contains the functions that do NOT return a regular
        # midas status code (as ctypes assumes functions return int by default).
        # For cm_transition, we have custom error-checking in the wrapper, so
        # specify it in return_types so the regular MidasError is not raised.
        self.return_types = {
            "c_cm_get_version": ctypes.c_char_p,
            "c_cm_get_revision": ctypes.c_char_p,
            "c_cm_transition": ctypes.c_int32,
            "c_free": None,
            "c_free_list": None,
            "c_rpc_is_remote": ctypes.c_int32
        }
        
        self.ignore_status_codes = {
            "c_cm_yield": ["SS_CLIENT_RECV", "SS_SERVER_RECV", "SS_TIMEOUT"],
            "c_bm_open_buffer": ["BM_CREATED"],
            "c_al_reset_alarm": ["AL_RESET"],
            "c_cm_exist": ["CM_NO_CLIENT"],
            "c_cm_shutdown": ["CM_NO_CLIENT"]
        }
        
        self.db_key_name_arg = {
            "c_db_find_key": 2,
            "c_db_get_value": 2
        }
    
    def __getitem__(self, name):
        """
        Overriding this function means we can register an error-checker for
        every function in the midas C library, without having to list them
        all manually.
        
        Args:
            * name (str) - Name of function in the C library.
        """
        
        # The base ctypes.CDLL class just returns a _FuncPtr, but we
        # automatically set the errcheck attribute to point to our
        # error-check function before returning it.
        func = self._FuncPtr((name, self))
        
        if not isinstance(name, int):
            func.__name__ = name
        
        if func.__name__ in self.return_types:
            # Functions that don't return int need to have
            # their return value set.
            func.restype = self.return_types[func.__name__]
        else:
            # Functions that do return int can have automatic
            # error-checking set up.
            func.errcheck = self.midas_errcheck
        
        return func
    
    def midas_errcheck(self, ret, func, args):
        """
        Raise an exception if the called function returned a non-SUCCESS status.
        This is called automatically by the ctypes framework.
        
        Args:
            * ret (int) - The function's return value
            * func (ctypes.CDLL._FuncPtr) - The function that was called
            * args (list) - The arguments passed to the function.
            
        Returns:
            int - The function's return value
            
        Raises:
            * See `status_code_to_exception()` documentation.
        """
        ignore_list = []
        keyname = None
        
        if hasattr(func, "__name__"):
            ignore_list = self.ignore_status_codes.get(func.__name__, [])
            keyname_index = self.db_key_name_arg.get(func.__name__, None)
            
            if keyname_index is not None and len(args) > keyname_index:
                if isinstance(args[keyname_index], ctypes.Array):
                    keyname = args[keyname_index].value.decode('ascii')
                elif isinstance(args[keyname_index], midas.structs.Key):
                    keyname = args[keyname_index].name.value.decode('ascii')
        
        # Convert from strings to ints if needed
        ignore_list = [status_codes.get(x, -1) if isinstance(x, str) else x for x in ignore_list]
                
        # This will raise an exception if needed
        status_code_to_exception(ret, ignore_list, keyname)
        
        return ret


def status_code_to_text(code):
    """
    Convert an integer midas status code to a string.
    
    Args:
        * code (int)
        
    Returns:
        str
    """
    return status_codes_to_text.get(code, "UNKNOWN_STATUS_CODE")
    
def status_code_to_exception(code, ignore, keyname=None):
    """
    Raise an exception based on the return code from a midas C function
    status code.
    
    Args:
        * code (int) - The function's return code
        * ignore (list of int) - Any return codes that should NOT cause an
            exception to be raised (in addition to "SUCCESS")
        * keyname (str) - If the status code was "DB_NO_KEY", which ODB path
            could not be found.
            
    Returns:
        None
        
    Raises:
        * KeyError if the code is "DB_NO_KEY"
        * TypeError if the code is "DB_TYPE_MISMATCH"
        * MidasError for any other code that isn't "SUCCESS"
    """
    if code in ignore:
        return
    if code == status_codes["SUCCESS"]:
        return
    if code == status_codes["DB_NO_KEY"]:
        if keyname is not None:
            msg = "Key '%s' not found in ODB" % keyname
        else:
            msg = "Key not found in ODB"
        raise KeyError(msg)
    if code == status_codes["DB_TYPE_MISMATCH"]:
        raise TypeError("Invalid type for ODB parameter")
    message = status_code_to_text(code)
    raise MidasError(code, message)
    

def safe_to_json(input_str, use_ordered_dict=False):
    """
    Convert input_str to a json structure, with arguments that will catch
    bad bytes.

    Args:

    * input_str (str)
    * as_ordered_dict (bool) - Whether to preserve the order of keys in the
        JSON document by using OrderedDict instead of dict.

    Returns:
        dict
    """
    decoded = input_str.decode("utf-8", "ignore")
    if use_ordered_dict:
        return json.loads(decoded, strict=False, object_pairs_hook=collections.OrderedDict)
    else:
        return json.loads(decoded, strict=False)
