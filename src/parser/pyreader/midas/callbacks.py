"""
This file contains tools to help set up python functions that will be called
when certain things happen (e.g. when a value changes in the ODB, or when a run
starts or stops.

They are used internally by functions in `midas.client.MidasClient` that
provide a more user-friendly interface (e.g. `odb_watch()`)
"""

import ctypes
import midas
import traceback
import json
import midas.event
import midas.structs
import datetime

# Callback functions we've instantiated for watching ODB records, keyed by ODB path
hotlink_callbacks = {}

# List of callback functions we've instantiated for watching ODB records where 
# callback cares about array index, keyed by ODB path
watch_callbacks = {}

# List of callback functions we've instantated for watching run transitions
transition_callbacks = []

# List of callback functions we've instantated for deferring a run transition
deferred_transition_callbacks = []

# List of callback functions we've instantated for RPC callbacks
rpc_callbacks = []

# List of callback functions we've instantiated for reading msg events
event_callbacks = []

# Callback function for when we disconnect from midas.
# Special as we handle this entirely in python, not in the C layer.
disconnect_callback = None

# Define the C style of the callback function we must pass to db_open_record
# return void; args int (hDB) / int (hKey) / void* (info; unused).
HOTLINK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

# Define the C style of the callback function we must pass to db_watch
# return void; args int (hDB) / int (hKey) / int (index) / void* (info; unused).
WATCH_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

# Define the C style of the callback function we must pass to cm_register_transition
# return int (status); args int (run number) / char* (error message)
TRANSITION_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char))

# Define the C style of the callback function we must pass to cm_register_deferred_transition
# return BOOL (start transition now?); args int (run number) / BOOL (first time being called?)
DEFERRED_TRANSITION_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)

# Define the C style of the callback function we must pass to cm_register_function
# return int (status); args int (index) / void** (params)
RPC_CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))

# Define the C style of the callback function we must pass to cm_msg_register
# return void; args int (buffer_handler) / int (request_id) / EVENT_HEADER* (event_header) / void* event_data
EVENT_HANDLER_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(midas.structs.EventHeader), ctypes.c_void_p)

def exception_message(e):
    """
    Get a printable version of an Exception.
    
    Args:
        e (Exception)
    Returns:
        str
    """
    if hasattr(e, 'message'):
        ret_str = e.message
    else:
        ret_str = str(e)
        
    if ret_str is None or len(ret_str) == 0:
        ret_str = type(e).__name__
        
    return ret_str


def make_hotlink_callback(path, callback, client):
    """
    Create a callback function that can be passed to db_open_record from the
    midas C library.
    
    Args:
        * path (str) - The ODB path that will be watched.
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Returns:
        * `HOTLINK_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * path (str) - The ODB path being watched
            * odb_value (float/int/dict etc) - The new ODB value
        * Value it should return:
            * Anything or nothing, we don't do anything with it 
    """
    
    # We create a closure that tracks the path being watched and the
    # client that created us. This means the python callback users 
    # provide don't have to worry about hDB/hKey etc, and can just
    # be told the new value etc.
    def _wrapper(hDB, hKey, info):
        odb_value = client.odb_get(path, recurse_dir=True)
        try:
            callback(client, path, odb_value)
        except Exception as e:
            traceback.print_exc()
            client.msg("Exception raised during callback on %s: %s" % (path, exception_message(e)), True)
        
    cb = HOTLINK_FUNC_TYPE(_wrapper)
    hotlink_callbacks[path] = cb
    return cb
    
def make_watch_callback(path, callback, client):
    """
    Create a callback function that can be passed to db_watch from the
    midas C library.
    
    Args:
        * path (str) - The ODB path that will be watched.
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Returns:
        * `WATCH_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * path (str) - The ODB path being watched
            * index (int/None) - Array index that changed if watching an array
            * odb_value (float/int/dict etc) - The new ODB value
        * Value it should return:
            * Anything or nothing, we don't do anything with it 
    """
    
    # We create a closure that tracks the path being watched and the
    # client that created us. This means the python callback users 
    # provide don't have to worry about hDB/hKey etc, and can just
    # be told the new value etc.
    def _wrapper(hDB, hKey, idx, info):
        changed_path = ""
        
        while True:
            k = client._odb_get_key_from_hkey(hKey)
            
            if k.parent_keylist == 0:
                # Found the /root entry
                break
            
            changed_path = "/" + k.name.decode("utf-8") + changed_path
            hKey = client._odb_get_parent_hkey(hKey)
        
        odb_value = client.odb_get(changed_path, recurse_dir=True)
        
        if isinstance(odb_value, list):
            odb_value = odb_value[idx]
        else:
            idx = None
        
        try:
            callback(client, changed_path, idx, odb_value)
        except Exception as e:
            traceback.print_exc()
            client.msg("Exception raised during callback on %s: %s" % (path, exception_message(e)), True)
        
    cb = WATCH_FUNC_TYPE(_wrapper)
    watch_callbacks[path] = cb
    return cb

def make_transition_callback(callback, client):
    """
    Create a callback function that can be passed to cm_register_transition 
    from the midas C library.
    
    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Returns:
        * `TRANSITION_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * run_number (int)
        * Value it should return:
            * None, int or 2-tuple of (int, str) - None or 1 indicate
                the transition was successful. If the transition was
                unsuccessful, the string can be used to indicate why
                (max length 255). Exceptions are caught and result in
                a status code of 605, with the string taken from the
                exception message.
    """
    
    # We create a closure that tracks which python callback function
    # to call, and also means users don't have to worry about filling
    # error_msg themselves.
    def _wrapper(run_number, error_msg):
        ret_int = None
        ret_str = None
        
        try:
            retval = callback(client, run_number)
        except Exception as e:
            traceback.print_exc()
            ret_str = exception_message(e)
            ret_int = midas.status_codes["FE_ERR_DRIVER"]
            retval = (ret_int, ret_str)
        
        if isinstance(retval, tuple) or isinstance(retval, list):
            ret_int = retval[0]
            if len(retval) > 1:
                ret_str = retval[1]
                
        if ret_int is None:
            ret_int = midas.status_codes["SUCCESS"]
            
        if not isinstance(ret_int, int):
            raise ValueError("Transition callback didn't return an allowed status (%s)" % ret_int)
            
        # Fill ret_str into error_msg (max 256 chars)
        if isinstance(ret_str, str) and len(ret_str) and error_msg:
            len_msg = min(len(ret_str), 255)
            for i in range(len_msg):
                error_msg[i] = ret_str[i].encode('ascii')
            error_msg[len_msg] = b'\x00'
            
        return ret_int 

    cb = TRANSITION_FUNC_TYPE(_wrapper)
    transition_callbacks.append(cb)
    return cb


def make_deferred_transition_callback(callback, client):
    """
    Create a callback function that can be passed to cm_register_deferred_transition 
    from the midas C library.
    
    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Returns:
        * `DEFERRED_TRANSITION_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * run_number (int)
        * Value it should return:
            * True if the transition can proceed, False if the transition
              should wait.
    """
    
    # We create a closure that tracks which python callback function
    # to call, and converts the python True/False into the BOOL that
    # midas expects. The dummy param is for midas' "first" parameter, which
    # doesn't seem to be very well documented.
    def _wrapper(run_number, dummy):
        try:
            retval = callback(client, run_number)
        except Exception as e:
            traceback.print_exc()
            client.msg("Exception raised during deferred transition callback: %s" % exception_message(e), True)
            retval = True

        return retval

    cb = DEFERRED_TRANSITION_FUNC_TYPE(_wrapper)
    deferred_transition_callbacks.append(cb)
    return cb


def make_rpc_callback(callback, client, return_success_even_on_failure=False):
    """
    Create a callback function that can be passed to cm_register_function from the midas
    C library.

    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        * return_success_even_on_failure (bool) - mjsonrpc (the web interface
            for calling JRPC functions) does not return any message if the
            status code isn't "SUCCESS". This can be annoying if you want to
            show a specific error message to the user, and not have them trawl
            through the midas message log. 
            If you set this parameter to False, then you get the "normal"
            behaviour, where the returned status code and result string are
            exactly what is returned from the callback function.
            If you set this parameter to True, then the status code will 
            always be "SUCCESS", and the result string will be JSON-encoded
            text of the form `{"code": 604, "msg": "Some error message"}.
        
    Returns:
        * `RPC_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * cmd (str) - The command user wants to execute
            * args (str) - Other arguments the user supplied
            * max_len (int) - The maximum string length the user accepts in the return value
        * Value it should return:
            * A tuple of (int, str) or just an int. The integer should be a status code
                from midas.status_codes. The string, if present, should be any text that
                should be returned to the caller. The maximum string length that will be
                returned to the user is given by the `max_len` parameter.
    """
    
    # As with the other make_xxx_callback functions, we create a closure.
    def _wrapper(index, params):
        cmd = ctypes.cast(params[0], ctypes.c_char_p).value.decode("utf-8")
        args = ctypes.cast(params[1], ctypes.c_char_p).value.decode("utf-8")
        buf_p = ctypes.cast(params[2], ctypes.POINTER(ctypes.c_char))
        max_reply_len = ctypes.cast(params[3], ctypes.POINTER(ctypes.c_int)).contents.value
        
        if max_reply_len <= 0:
            retval = midas.status_codes["FE_ERR_DRIVER"]
            client.msg("max_reply_len must be > 0 when calling JRPC function", True)
        else:
            try:
                retval = callback(client, cmd, args, max_reply_len)
            except Exception as e:
                traceback.print_exc()
                retval = (midas.status_codes["FE_ERR_DRIVER"], exception_message(e))

        if isinstance(retval, int):
            ret_int = retval
            ret_str = ""
        elif retval is None or len(retval) != 2:
            ret_int = midas.status_codes["FE_ERR_DRIVER"]
            ret_str = "Invalid return value from callback functions"
        else:
            (ret_int, ret_str) = retval
            
        if return_success_even_on_failure:
            # Encode result, ensuring our JSON struct will be valid
            temp = {"code": ret_int, "msg": ""}
            max_msg_len = max_reply_len - 1 - len(json.dumps(temp))
            
            if max_msg_len < 0 and max_reply_len > 0:
                # Give up - buffer is way too small
                ret_int = midas.status_codes["FE_ERR_DRIVER"]
                ret_str = "Return buffer size too small for JSON-encoded result"
            else:
                # Encode the result
                temp["msg"] = ret_str[:max_msg_len]
                ret_int = midas.status_codes["SUCCESS"]
                ret_str = json.dumps(temp)
        
        # Write return value to buffer midas created for us.
        addr = ctypes.addressof(buf_p.contents)
        dest_chars = (ctypes.c_char * max_reply_len).from_address(addr)
        write_size = min(len(ret_str), max_reply_len - 1)
        
        if write_size > 1:
            dest_chars[:write_size] = bytes(ret_str, 'utf-8')[:write_size]
            dest_chars[write_size] = b'\0'
        
        if ret_int != midas.status_codes["SUCCESS"] and len(ret_str):
            # mjsonrpc only returns the string to the user if status is
            # SUCCESS. Make sure we msg the error so user knows what the
            # problem was.
            client.msg("Error running RPC function: %s" % ret_str, True)
        
        return ret_int

    cb = RPC_CALLBACK_FUNC_TYPE(_wrapper)
    rpc_callbacks.append(cb)
    return cb


def make_brpc_callback(callback, client):
    """
    Create a callback function that can be passed to cm_register_function from the midas
    C library.

    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Returns:
        * `RPC_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * cmd (str) - The command user wants to execute
            * args (str) - Other arguments the user supplied
            * max_len (int) - The maximum string length the user accepts in the return value
        * Value it should return:
            * A tuple of (int, `ctypes.Array of ctypes.c_char`) or (int, `midas.event.Event`).
              In the first version you can return arbiotrary binary data. Best to create the
              relevant object using `ctypes.create_string_buffer()`.
              In the second version we will automatically convert the event into the relevant
              binary representation.
              The maximum data length (in bytes) that will be returned to the user is given by the 
              `max_len` parameter.
    """
    
    # As with the other make_xxx_callback functions, we create a closure.
    def _wrapper(index, params):
        cmd = ctypes.cast(params[0], ctypes.c_char_p).value.decode("utf-8")
        args = ctypes.cast(params[1], ctypes.c_char_p).value.decode("utf-8")
        buf_p = ctypes.cast(params[2], ctypes.POINTER(ctypes.c_char))
        max_reply_len_p = ctypes.cast(params[3], ctypes.POINTER(ctypes.c_int))
        max_reply_len = max_reply_len_p.contents.value
        
        if max_reply_len <= 0:
            retval = midas.status_codes["FE_ERR_DRIVER"]
            client.msg("max_reply_len must be > 0 when calling BRPC function", True)
        else:
            try:
                retval = callback(client, cmd, args, max_reply_len)
            except Exception as e:
                traceback.print_exc()
                retval = (midas.status_codes["FE_ERR_DRIVER"], exception_message(e))

        if isinstance(retval, int):
            ret_int = retval
            ret_data = ctypes.create_string_buffer(b'\0')
        elif retval is None or len(retval) != 2:
            ret_int = midas.status_codes["FE_ERR_DRIVER"]
            ret_data = ctypes.create_string_buffer(b'\0')
        else:
            (ret_int, ret_data) = retval

        if isinstance(ret_data, midas.event.Event):
            # These events aren't going to end up in a midas buffer,
            # so we don't care if the user forgot to specify the event
            # ID and serial number. We'll just set dummy values for them.
            if ret_data.header.event_id is None:
                ret_data.header.event_id = 12345
            if ret_data.header.serial_number is None:
                ret_data.header.serial_number = 0

            ret_data = ret_data.pack()

        if not isinstance(ret_data, ctypes.Array) or ret_data._type_ != ctypes.c_char:
            print("Invalid data type returned from BRPC callback (%s)" % type(ret_data))
            ret_int = midas.status_codes["FE_ERR_DRIVER"]
            ret_data = ctypes.create_string_buffer(b'\0')
        
        # Write return value to buffer midas created for us.
        addr = ctypes.addressof(buf_p.contents)
        dest_chars = (ctypes.c_char * max_reply_len).from_address(addr)
        write_size = min(len(ret_data), max_reply_len - 1)
        
        if write_size > 1:
            dest_chars[:write_size] = ret_data[:write_size]
            dest_chars[write_size] = b'\0'

        max_reply_len_p.contents.value = write_size
        
        return ret_int

    cb = RPC_CALLBACK_FUNC_TYPE(_wrapper)
    rpc_callbacks.append(cb)
    return cb


def make_msg_handler_callback(callback, client, only_message_types):
    """
    Create a callback function that can be passed to cm_msg_register 
    from the midas C library.
    
    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        * only_message_types (None, or list of midas.MT_xxx flags) - which message types
            to pass to the callback function (e.g. only midas.MT_ERROR, not midas.MT_INFO)
        
    Returns:
        * `EVENT_HANDLER_FUNC_TYPE`
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
            * msg (str) - the actual message
            * timestamp (datetime.datetime) - the timestamp of the message
            * msg_type (int) - midas.MT_xxx flag, e.g. midas.MT_ERROR
        * Value it should return:
            * Anything or nothing, we don't do anything with it 
    """
    
    # We create a closure that tracks which python callback function
    # to call and which message types to pass on.
    def _wrapper(buffer_handler, request_id, event_header_p, event_data_p):
        try:
            event_header = event_header_p.contents
            timestamp = datetime.datetime.fromtimestamp(event_header.time_stamp)
            msg_type = event_header.trigger_mask

            if only_message_types is not None and msg_type not in only_message_types:
                return
                
            msg = ctypes.cast(event_data_p, ctypes.c_char_p).value.decode("utf-8")
            callback(client, msg, timestamp, msg_type)
        except Exception as e:
            traceback.print_exc()
            client.msg("Exception raised during message handler callback: %s" % exception_message(e), True)

        return

    cb = EVENT_HANDLER_FUNC_TYPE(_wrapper)
    event_callbacks.append(cb)
    return cb


def make_and_store_disconnect_callback(callback, client):
    """
    Create a callback function that we will TRY to call when we disconnect
    from midas.
    
    Args:
        * callback (function) - See below.
        * client (midas.client.MidasClient) - the client that's creating this.
        
    Python function (`callback`) details:
        * Arguments it should accept:
            * client (midas.client.MidasClient)
        * Value it should return:
            * Anything or nothing
    """
    global disconnect_callback
    
    # We create a closure that tracks which python callback function
    # to call, and pass the client instance to it.
    def _wrapper():
        try:
            callback(client)
        except Exception as e:
            traceback.print_exc()
            client.msg("Exception raised during disconnect callback: %s" % exception_message(e), True)
            
    if callback is None:
        disconnect_callback = None
    else:
        disconnect_callback = _wrapper