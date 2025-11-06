"""
This module contains user-friendly pythonic wrappers around the midas C 
library.
"""

import midas
import midas.structs
import midas.callbacks
import midas.event
import ctypes
import os
import os.path
import sys
import datetime
import inspect
import collections
import logging
import time

try:
    import numpy as np
    have_numpy = True
except:
    have_numpy = False

ver = sys.version_info
if ver <= (3, 0):
    raise EnvironmentError("Sorry, we only support python 3, but you're running version %d.%d" % (ver[0], ver[1]))

_midas_connected = False # Whether we've connected to a midas experiment
_midas_lib_loaded = None # The `midas.MidasLib` we have loaded

logger = logging.getLogger("midas")

class MidasClient:
    """
    This is the main class that contains pythonic wrappers around the midas
    C library. Most users will interact with this class, the definitions in
    the main midas.__init__ file, and optionally the Frontend class defined
    in midas.frontend.
    
    # Normal access
    
    Most common functions have wrappers defined in this class. For example,
    the `odb_set` function allows you to pass a normal python values as
    arguments:
    
    ```
    import midas.client
    
    client = midas.client.MidasClient("NormalExample")
    client.odb_set("/my/odb_path", 4.56)
    ```
    
    # Low-level access
    
    If you want to use a function that doesn't have a python wrapper, you will
    need to:
    
    * Define an appropriate C-compatible function in include/midas_c_compat.h
        and src/midas_c_compat.cxx
    * Recompile using cmake (cd build && make && make install)
    * Call the function from python like client.lib.c_my_func(). You will need 
        to convert any arguments to ctypes values (and if the C function 
        doesn't return an integer, then there is even more work to do - see the
        MidasLib class in the main `midas` module).
    
    # Members
    
    * name (str) - The name of this client.
    * lib (`midas.MidasLib`) - The ctypes wrapper of the midas library.
    * hDB (`ctypes.c_int`) - HNDLE to the experiment's ODB
    * hClient (`ctypes.c_int`) - HNDLE to this client's entry in the ODB
    * event_buffers (dict of {int: `ctypes.c_char_p`}) - Character buffers that
        have been opened, keyed by buffer handle.
    """
    def __init__(self, client_name, host_name=None, expt_name=None, daemon_flag=None, throw_if_already_running=False):
        """
        Load the midas library, and connect to the chosen experiment.
        
        Args:
            * client_name (str) - The name this program will be referred to
                within the midas system (e.g. on the "Programs" webpage).
            * host_name (str) - Which server to connect to. If None, it will
                be determined from the MIDAS_SERVER_HOST environment variable
                (and will default to this machine if MIDAS_SERVER_HOST is not
                set).
            *  expt_name (str) - Which experiment to connect to on the server.
                If None, it will be determined automatically from the 
                MIDAS_EXPT_NAME environment variable.
            * daemon_flag (None/int) - None means do not deamonize this process; 
                0 means deamonize and close stdout; 1 means deamonize and keep stdout.
            * throw_if_already_running (bool) - If True, and a program called `client_name`
                is already connected to midas, we will raise a RuntimeError exception.
        """
        global _midas_connected, _midas_lib_loaded
        
        if _midas_connected:
            raise RuntimeError("You can only have one client connected to midas per process! Disconnect any previous clients you created before creating a new one.")
        
        if not _midas_lib_loaded:
            midas_sys = os.getenv("MIDASSYS", None)
            
            if midas_sys is None:
                raise EnvironmentError("Set environment variable $MIDASSYS to path of midas")
            
            lib_files = ["libmidas-c-compat.so", "libmidas-c-compat.dylib"]
            lib_dir = os.path.join(midas_sys, "lib")
            lib_path = None
            
            for lib_file in lib_files:
                test = os.path.join(lib_dir, lib_file)
                
                if os.path.exists(test):
                    lib_path = test
                    break
                
            if lib_path is None:
                raise EnvironmentError("Couldn't find libraries %s in %s - make sure midas was compiled using cmake" % (lib_files, lib_path))
            
            # Automatically checks return codes and raises exceptions.
            _midas_lib_loaded = midas.MidasLib(lib_path)
            
        self.name = client_name    
        self.lib = _midas_lib_loaded
        
        # Deamonize before we connect, or else the buffer handles etc will be associated with the wrong PID
        if daemon_flag is not None:
            self.lib.c_ss_daemon_init(daemon_flag)
        
        c_host_name = ctypes.create_string_buffer(256)
        c_expt_name = ctypes.create_string_buffer(32)
        c_client_name = ctypes.create_string_buffer(client_name.encode('ascii'), 32)
        
        if host_name is None or expt_name is None:
            self.lib.c_cm_get_environment(c_host_name, ctypes.sizeof(c_host_name), c_expt_name, ctypes.sizeof(c_expt_name))
            
        if host_name is not None:
            c_host_name.value = host_name.encode('ascii')
            
        if expt_name is not None:
            c_expt_name.value = expt_name.encode('ascii')
        
        # We automatically connect to this experiment's ODB
        self.lib.c_cm_connect_experiment(c_host_name, c_expt_name, c_client_name, None)
        _midas_connected = True

        if throw_if_already_running and self.client_exists(client_name):
            raise RuntimeError(f"{client_name} is already running")
                
        self.hDB = ctypes.c_int()
        self.hClient = ctypes.c_int()

        self.lib.c_cm_get_experiment_database(ctypes.byref(self.hDB), ctypes.byref(self.hClient))
        self.lib.c_cm_start_watchdog_thread()
        
        self.event_buffers = {}
        
    def disconnect(self):
        """
        Nicely disconnect from midas. If this function isn't called at the end
        of a program, midas will complain in the message log and it might take
        a while for our disappearance to be noted.
        
        This function will be called automatically from the destructor of this
        class, HOWEVER some implementations of python do not guarantee that
        __del__ will be called at system exit.
        
        To be sure of a trouble-free disconnection from midas, you should either:
        * Call this function at the end of your program. 
        * Wrap your usage in a `with` statement:
            `with midas.client.MidasClient('name') as c: .......`
        """
        global _midas_connected
        
        if _midas_connected and hasattr(self, 'lib') and self.lib:
            if midas.callbacks.disconnect_callback is not None:
                midas.callbacks.disconnect_callback()

            self.lib.c_cm_disconnect_experiment()
            
            # We do a few extra things here that are unusual for regular midas clients.
            # Mostly for unit tests where we create and delete many midas connections in
            # the same process. The biggest issue is when we register JRPC/BRPC callbacks
            # and thus become an RPC server (whereas most midas programs are only ever
            # RPC clients). cm_disconnect_experiment() isn't sufficient to cleanup all those
            # server-style resources, so we must call these 2 extra functions as well.
            # Otherwise when the next test creates a new client, it will get confused by
            # existing entries in _ss_server_acceptions etc.
            self.lib.c_rpc_server_shutdown()
            self.lib.c_ss_suspend_reset_server_acceptions()

            _midas_connected = False
            
    def __del__(self):
        """
        Python doesn't guarantee that __del__ will be called at the end of a
        program, but if we do get called we should disconnect from midas.
        """
        self.disconnect()
    
    def __enter__(self):
        """
        For context manager usage (`with`).
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Guaranteed to disconnect from midas cleanly when using a context
        manager style (`with` syntax).
        """
        self.disconnect()

    def msg(self, message, is_error=False, facility="midas"):
        """
        Send a message into the midas message log.
        
        These messages are stored in a text file, and are visible on the the
        "messages" webpage.
        
        Args:
            * message (str) - The actual message.
            * is_error (bool) - Whether this message is informational or an 
                error message. Error messages are highlighted in red on the
                message page.
            * facility (str) - The log file to write to. Vast majority of
                messages are written to the "midas" facility.
        """
        
        # Find out where this function was called from. We go up
        # 1 frame in the stack to get to the lowest-level user
        # function that called us.
        # 0. fn_A()
        # 1. fn_B() # <--- Get this function
        # 2. midas.client.msg()
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        filename = ctypes.create_string_buffer(bytes(caller.filename, "utf-8"))
        line = ctypes.c_int(caller.lineno)
        routine = ctypes.create_string_buffer(bytes(caller.function, "utf-8"))
        c_facility = ctypes.create_string_buffer(bytes(facility, "utf-8"))
        c_msg = ctypes.create_string_buffer(bytes(message, "utf-8"))
        msg_type = ctypes.c_int(midas.MT_ERROR if is_error else midas.MT_INFO)
    
        self.lib.c_cm_msg(msg_type, filename, line, c_facility, routine, c_msg)

    def communicate(self, time_ms, return_immediately_if_work_done=False):
        """
        If you have a long-running script, this function should be called
        repeatedly in a loop. It serves several purposes:
        * letting midas know the client is still working
        * allowing users to stop your program from the "Programs" webpage
        * telling your client about any ODB changes that it opened a hotlink for
        * letting your client know about run transitions
        * and more...
        
        Args:
            * time_ms (int) - How many millisecs to wait before returning
                from this function. If negative, will block until program
                is told to shutdown.
            * return_immediately_if_work_done (bool) - Return immediately
                if we did some work(). By default we'll keep waiting for
                more work until the full user-specified time has expired.
        """
        first = True
        c_time_ms = ctypes.c_int()
        start = datetime.datetime.now()
        
        # cm_yield may return early if some events were available, so we run
        # in a loop until the timeout the user specified has expired.
        while first or time_ms > 0:
            first = False
            c_time_ms.value = time_ms
            
            try:
                self.lib.c_cm_yield(c_time_ms)
            except midas.MidasError as e:
                if e.code == midas.status_codes["RPC_SHUTDOWN"]:
                    self.disconnect()
                    print("Midas shutdown")
                    exit(0)
                else:
                    raise
                
            now = datetime.datetime.now()
            elapsed_ms = int((now - start).total_seconds() * 1000) + 1 # Avoid slight rounding errors
            time_ms -= elapsed_ms

            if return_immediately_if_work_done:
                break
    
    def odb_exists(self, path):
        """
        Whether the given path exists in the ODB.
        
        Args:
            * path (str) - e.g. "/Runinfo/State"
            
        Returns:
            bool
        """
        path = self._odb_strip_final_slash(path)
        c_path = ctypes.create_string_buffer(bytes(path, "utf-8"))
        hKey = ctypes.c_int()
        
        try:
            self.lib.c_db_find_key(self.hDB, 0, c_path, ctypes.byref(hKey))
        except KeyError:
            return False
        
        return True
    
    def odb_delete(self, path, follow_links=False):
        """
        Delete the given location in the ODB (and all of its children if the
        specified path points to a directory).
        
        Args:
            * path (str) - e.g. "/Runinfo/State"
            * follow_links (bool) - Whether to follow symlinks in the ODB or 
                not. If False, we'll delete the link itself, rather than the 
                target of the link.
        """
        path = self._odb_strip_final_slash(path)
        c_path = ctypes.create_string_buffer(bytes(path, "utf-8"))
        c_follow = ctypes.c_uint32(follow_links)
        hKey = ctypes.c_int()
        
        if follow_links:
            self.lib.c_db_find_key(self.hDB, 0, c_path, ctypes.byref(hKey))
        else:
            self.lib.c_db_find_link(self.hDB, 0, c_path, ctypes.byref(hKey))
            
        self.lib.c_db_delete_key(self.hDB, hKey, c_follow)
        
    def odb_get(self, path, recurse_dir=True, include_key_metadata=False, just_key_list=False):
        """
        Get the content of an ODB entry, with automatic conversion to appropriate
        python types. You can request either a single ODB key or an entire
        directory (in which case a dict will be returned).
        
        A note on dicts - midas directories have an associated order to them
        (you can rearrange the order of entries in a midas directory by calling
        `db_reorder_key` in the C library). Until python3.7, basic python dicts
        did not have a well-defined order. Therefore we return a 
        `collections.OrderedDict` rather than a basic dict, so the order in 
        which you iterate over the keys will match the order in which you 
        see them on the midas status page, for example.
        
        Args:
            * path (str) - e.g. "/Runinfo". You may specify a single array index
                if desired (e.g. "/Path/To/My/Array[1]").
            * recurse_dir (bool) - If `path` points to a directory, whether to
                recruse down and get all entries beneath that directory, or just
                the entries immediately below the specified path. `recurse_dir`
                being False acts more like an `ls` of the directory, while it
                being True acts more like a full dump of the content.
            * include_key_metadata (bool) - Whether to include extra information
                in the return value containing metadata about each ODB key. 
                Examples metadata include when the ODB key was last written, the
                data type (e.g. TID_INT) etc. The metadata is stored in extra
                dict entries named as "<name>/key" (e.g. if the actual ODB 
                content is in "State", the metadata is in "State/key").
            * just_key_list (bool) - Just "ls" an ODB directory - don't recurse,
                don't return values - just list the key names.
                
        Returns:
            * If `path` points to a directory, a dict. Dicts are keyed by the
                ODB key name. If `recurse_dir` is False and one of the entries
                in `path` is also a directory, it's value will be an empty dict.
                If `include_key_metadata` is True, the keys at the top level will
                be the "directory_name" and "directory_name/key"
            * If `path` points to a single ODB key, an int/float/bool etc (or
                a list of them). If `include_key_metadata` is True the return 
                value will be a dict, with two entries (one for the value and
                one for the metadata).
            * If `just_key_list` is True, and `path` points to a directory, a list
                of strings.
                
        Raises:
            * KeyError if `path` doesn't exist in the ODB
            * midas.MidasError for other midas-related issues
            * ValueError if `just_key_list` is True and `path` isn't a directory.
        """
        path = self._odb_strip_final_slash(path)
        (array_single, dummy, dummy2) = self._odb_path_is_for_single_array_element(path)
        c_path = ctypes.create_string_buffer(bytes(path, "utf-8"))
        
        hKey = ctypes.c_int()
        self.lib.c_db_find_key(self.hDB, 0, c_path, ctypes.byref(hKey))
        
        key_metadata = self._odb_get_key_from_hkey(hKey)

        if just_key_list:
            if key_metadata.type != midas.TID_KEY:
                raise ValueError(f"ODB entry '{path}' is not a directory, but `just_key_list` was specified")
        
            data = self._odb_enum_key(hKey, False)
            return [d[1].name.decode("utf-8") for d in data]
        
        if key_metadata.type == midas.TID_KEY and not self._rpc_is_remote():
            # If we are getting the content of a directory, we can use the simple
            # db_copy_json_xxx functions. These only work if we're talking to a 
            # local midas experiment though (not via RPC).
            buf = ctypes.c_char_p()
            bufsize = ctypes.c_int()
            bufend = ctypes.c_int()

            if recurse_dir:
                self.lib.c_db_copy_json_save(self.hDB, hKey, ctypes.byref(buf), ctypes.byref(bufsize), ctypes.byref(bufend))
            else:
                self.lib.c_db_copy_json_ls(self.hDB, hKey, ctypes.byref(buf), ctypes.byref(bufsize), ctypes.byref(bufend))
    
            if bufsize.value > 0:
                retval = midas.safe_to_json(buf.value, use_ordered_dict=True)
            else:
                retval = {}
        
            retval = self._convert_dwords_from_odb(retval)
        
            if include_key_metadata:
                # Subkeys already include metadata, we just need to add
                # the metadata for the top-level key.
                name_data = "%s" % key_metadata.name.decode("utf-8")
                name_meta = "%s/key" % name_data
                retval = {name_data: retval}
                retval[name_meta] = self._odb_key_metadata_to_dict(key_metadata)
            else:
                # Prune metadata from the subkeys.
                retval = self._prune_metadata_from_odb(retval)
                
            # Free the memory allocated by midas
            self.lib.c_free(buf)
        else:
            # Get the value using db_get_value etc, recursing ourselves
            # if needed.
            retval = self._odb_get_value(path, 
                                         recurse_dir=recurse_dir, 
                                         array_single=array_single, 
                                         include_key_metadata=include_key_metadata, 
                                         key_metadata=key_metadata,
                                         hKey=hKey)
            
        return retval
    
    def odb_set(self, path, contents, create_if_needed=True, 
                    remove_unspecified_keys=True, resize_arrays=True, 
                    lengthen_strings=True, explicit_new_midas_type=None, 
                    update_structure_only=False):
        """
        Set the value of an ODB key, or an entire directory. You may pass in
        normal python values, lists and dicts and they will be converted to
        appropriate midas ODB key types (e.g. int becomes midas.TID_INT, bool
        becomes midas.TID_BOOL).
        
        Sensible defaults have been chosen for converting python types to the C 
        types used internally in the midas ODB. However if you want more control 
        over the ODB type, you may use the types defined in the ctypes library.
        For example, regular python integers become a midas.TID_INT, but you can
        use a `ctypes.c_uint32` to get a midas.TID_DWORD.
        
        If you are setting the content of a directory and care about the order
        in which the entries appear in that directory, `contents` should be a
        `collections.OrderedDict` rather than a basic python dict. See the note
        in `odb_get` for more about dictionary ordering.
        
        Args:
            * path (str) - The ODB entry to set. You may specify a single array
                index if desired (e.g. "/Path/To/My/Array[1]").
            * contents (int/float/string/tuple/list/dict etc) - The new value to set
            * create_if_needed (bool) - Automatically create the ODB entry
                (and parent directories) if needed.
            * remove_unspecified_keys (bool) - If `path` points to a directory
                and `contents` is a dict, remove any ODB keys within `path` that
                aren't present in `contents`. This means that the ODB will exactly
                match the dict you passed in. You may want to set this to False
                if you want to only update a few entries within a directory, and
                want to do so with only a single call to `odb_set()`.
            * resize_arrays (bool) - Automatically resize any ODB arrays to match 
                the length of lists present in `contents`. Arrays will be both
                lengthened and shortened as needed.
            * lengthen_strings (bool) - Automatically increase the storage size
                of a TID_STRING entry in the ODB if it is not long enough to 
                store the value specified. We will include enough space for a
                final null byte.
            *  explicit_new_midas_type (one of midas.TID_xxx) - If you're 
                setting the value of a single ODB entry, you can explicitly
                specify the type to use when creating the ODB entry (if needed).
            * update_structure_only (bool) - If you want to add/remove entries
                in an ODB directory, but not change the value of any entries
                that already exist. Only makes sense if contents is a dict / 
                `collections.OrderedDict`. Think of it like db_check_record from
                the C library.
                
        Raises:
            * KeyError if `create_if_needed` is False and the ODB entry does not
                already exist.
            * TypeError if there is a problem converting `contents` to the C 
                type we must pass to the midas library (e.g. the ODB entry is
                a TID_STRING but you passed in a float).
            * ValueError if there is a non-type-related problem with `contents`
                (e.g. if `resize_arrays` is False and you provided a list that
                doesn't match the size of the existing ODB array).
            * midas.MidasError if there is some other midas issue.
        """
        path = self._odb_strip_final_slash(path)
        did_create = False
        
        (array_single, array_index, c_path_no_idx) = self._odb_path_is_for_single_array_element(path)
        
        if not self.odb_exists(path):
            if create_if_needed:
                if explicit_new_midas_type:
                    # User told use what ODB type to use.
                    new_midas_type = explicit_new_midas_type
                else:
                    # Choose ODB type based on input (e.g. a dict means
                    # we'll create a TID_KEY, an int means TID_INT etc)
                    new_midas_type = self._ctype_to_midas_type(contents)
                    
                self.lib.c_db_create_key(self.hDB, 0, c_path_no_idx, new_midas_type)
                did_create = True
            else:
                raise KeyError("ODB path doesn't exist")

        hKey = ctypes.c_int()
        self.lib.c_db_find_key(self.hDB, 0, c_path_no_idx, ctypes.byref(hKey))
        key_metadata = self._odb_get_key_from_hkey(hKey)
                
        if key_metadata.type == midas.TID_KEY:
            # This is currently inefficient and can result in many rather slow
            # calls to ctypes. It would be nicer to use db_paste_json, but that
            # doesn't allow us to have the remove_unspecified_keys=False mode.
            if not isinstance(contents, dict):
                raise TypeError("Pass a dict when setting the content of a midas directory")
            
            if not path.endswith("/"):
                path += "/"

            if remove_unspecified_keys:
                current_keys = self.odb_get(path, just_key_list=True)
                user_keys = frozenset(contents.keys())

                for k in current_keys:
                    if k not in user_keys:
                        self.odb_delete(path + k)
                    
            for k,v in contents.items():
                update = did_create or isinstance(v, dict) or not update_structure_only or not self.odb_exists(path + k)
                
                if update:
                    self.odb_set(path + k, v, create_if_needed, remove_unspecified_keys, resize_arrays, lengthen_strings, update_structure_only=update_structure_only)

            if isinstance(contents, collections.OrderedDict):
                self._odb_fix_directory_order(path, hKey, contents)
        else:
            # Single value
            if key_metadata.type == midas.TID_STRING and (lengthen_strings or did_create):
                changed = self._odb_lengthen_string_to_fit_content(key_metadata, c_path_no_idx, contents)
                if changed:
                    key_metadata = self._odb_get_key_from_hkey(hKey)
                
            if resize_arrays or did_create:
                changed = self._odb_resize_array_to_match_content(key_metadata, hKey, contents, array_index)
                if changed:
                    key_metadata = self._odb_get_key_from_hkey(hKey)

            if key_metadata.type == midas.TID_STRING:
                str_len = key_metadata.item_size
            else:
                str_len = None

            c_value = self._midas_type_to_ctype(key_metadata.type, array_len=None, str_len=str_len, initial_value=contents, odb_path=path)
            our_num_values = self._odb_get_content_array_len(key_metadata, contents)
            
            if array_single:
                # Single value in an existing array
                if key_metadata.type == midas.TID_STRING:
                    using_str_len = ctypes.sizeof(c_value)
                    odb_str_len = key_metadata.item_size
                    
                    if using_str_len != odb_str_len:
                        raise ValueError("ODB string length is %s, but we want to set strings of length %s" % (odb_str_len, using_str_len))

                self.lib.c_db_set_value_index(self.hDB, 0, c_path_no_idx, ctypes.byref(c_value), ctypes.sizeof(c_value), array_index, key_metadata.type, 0)
            else:
                # Single value for a single value
                if key_metadata.num_values != our_num_values:
                    raise ValueError("ODB array length is %s, but you provided %s values" % (key_metadata.num_values, our_num_values))
                
                if key_metadata.type == midas.TID_STRING:
                    using_str_len = ctypes.sizeof(c_value) / key_metadata.num_values
                    odb_str_len = key_metadata.item_size
                    
                    if using_str_len != odb_str_len:
                        raise ValueError("ODB string length is %s, but we want to set strings of length %s" % (odb_str_len, using_str_len))

                self.lib.c_db_set_data(self.hDB, hKey, ctypes.byref(c_value), ctypes.sizeof(c_value), our_num_values, key_metadata.type)
    
    def odb_link(self, link_path, destination_path):
        """
        Create or edit a link within the ODB.
        
        To remove the link call `odb_delete` with the `follow_links` flag set 
        to False.
        
        Args:
            * link_path (str) - The ODB path for the link name
            * destination_path (str) - Where in the ODB the link points to
            
        Raises:
            * ValueError if link_path already exists, but isn't already a link
        """
        link_path = self._odb_strip_final_slash(link_path)
        destination_path = self._odb_strip_final_slash(destination_path)
        c_link_path = ctypes.create_string_buffer(bytes(link_path, "utf-8"))
        c_dest_path = ctypes.create_string_buffer(bytes(destination_path, "utf-8"))
        
        current_handle = None
        current_key = None
        
        try:
            current_handle = self._odb_get_hkey(link_path, False)
        except KeyError:
            pass
        
        if current_handle:
            current_key = self._odb_get_key_from_hkey(current_handle)
            
            if current_key.type != midas.TID_LINK:
                raise ValueError("Path '%s' already exists, but isn't a link" % link_path)
            
            c_size = ctypes.c_int(ctypes.sizeof(c_dest_path))
            self.lib.c_db_set_link_data(self.hDB, current_handle, c_dest_path, c_size, 1, midas.TID_LINK)
        else:
            self.lib.c_db_create_link(self.hDB, 0, c_link_path, c_dest_path)
        
    def odb_get_link_destination(self, link_path):
        """
        Get where in the ODB a link points to.
        
        Args:
            * link_path (str) - The ODB path for the link name        
        
        Returns:
            str, the ODB path the link points to
        """
        link_path = self._odb_strip_final_slash(link_path)
        c_dest_path = ctypes.create_string_buffer(256)
        c_size = ctypes.c_int(ctypes.sizeof(c_dest_path))
        hKey = self._odb_get_hkey(link_path, follow_link=False)
        self.lib.c_db_get_link_data(self.hDB, hKey, c_dest_path, ctypes.byref(c_size), midas.TID_LINK)
        return c_dest_path.value.decode("utf-8")
    
    def odb_rename(self, current_path, new_name):
        """
        Rename an existing ODB entry. This function does not allow you to change
        which directory the entry is in, just the name of it.
        
        Args:
            * current_path (str) - Full ODB path of an existing ODB entry.
            * new_name (str) - The new name of the entry (do not include any
                `/` characters; the directory cannot be changed).
        """
        current_path = self._odb_strip_final_slash(current_path)
        hKey = self._odb_get_hkey(current_path)
        c_new_name = ctypes.create_string_buffer(bytes(new_name, "utf-8"))
        self.lib.c_db_rename_key(self.hDB, hKey, c_new_name)
    
    def odb_last_update_time(self, path):
        """
        Get when an ODB key was last written to.
        
        Args:
            * path (str) - The ODB path
            
        Returns:
            datetime.datetime
        """
        path = self._odb_strip_final_slash(path)
        ts = self._odb_get_key(path).last_written
        return datetime.datetime.fromtimestamp(ts)
    
    def odb_watch(self, path, callback, pass_changed_value_only=False):
        """
        Register a function that will be called when a value in the ODB changes.
        You must later call `communicate()` so that midas can inform your
        client about any ODB changes (if you're using the `midas.frontend`
        framework, then `communicate()` will be called for you automatically).
        
        Args:
            * path (str) - The ODB path to watch (including all its children).
            * callback (function) - See below. You can register the same 
                callback function to watch multiple ODB entries if you wish.
            * pass_changed_value_only (bool) - Whether your callback function
                expects to be passed:
                - True: the full ODB entry that is being watched
                - False: only the value that has changed
            
        Python function (`callback`) details if pass_changed_value_only is FALSE:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
                * path (str) - The ODB path being watched
                * odb_value (float/int/dict etc) - The new ODB state of the path being watched
            * Value it should return:
                * Anything or nothing, we don't do anything with it 
            
        Python function (`callback`) details if pass_changed_value_only is TRUE:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
                * path (str) - The ODB path that changed (possibly a child if you're watching
                    an entire ODB directory)
                * index (int/None) - The array index that changed (or None if the entry that changed
                    is not an array)
                * odb_value (float/int/dict etc) - The new ODB value (the single element
                    that changed if you're watching an array)
            * Value it should return:
                * Anything or nothing, we don't do anything with it 
                
        Example (see `examples/basic_client.py` for more:
        
        ```
        import midas.client
        
        def my_odb_callback(client, path, odb_value):
            print("New ODB content at %s is %s" % (path, odb_value))

        my_client = midas.client.MidasClient("pytest")
        
        # Note we pass in a reference to the function - we don't call it!
        # (i.e. we pass `my_odb_callback` not `my_odb_callback()`).
        my_client.odb_watch("/Runinfo", my_odb_callback)
        
        while True:
            my_client.communicate(100)
        ```
        """
        path = self._odb_strip_final_slash(path)
        logger.debug("Registering callback function for watching %s" % path)
        hKey = self._odb_get_hkey(path)
        
        if pass_changed_value_only:
            cb = midas.callbacks.make_watch_callback(path, callback, self)
            self.lib.c_db_watch(self.hDB, hKey, cb, None)
        else:
            cb = midas.callbacks.make_hotlink_callback(path, callback, self)
            self.lib.c_db_open_record(self.hDB, hKey, None, 0, 1, cb, None)
        
    def odb_stop_watching(self, path):
        """
        Stop watching any ODB entries that you previously registered through
        `odb_watch()`.
        
        Args:
            * path (str) - The ODB path you no longer want to watch.
        """
        path = self._odb_strip_final_slash(path)
        logger.debug("De-registering callback function that was watching %s" % path)
        hKey = self._odb_get_hkey(path)
        
        if path in midas.callbacks.watch_callbacks:
            self.lib.c_db_unwatch(self.hDB, hKey)
            del midas.callbacks.watch_callbacks[path]
        elif path in midas.callbacks.hotlink_callbacks:
            self.lib.c_db_close_record(self.hDB, hKey)
            del midas.callbacks.hotlink_callbacks[path]
        
    def open_event_buffer(self, buffer_name, buf_size=None, max_event_size=None):
        """
        Open a buffer that can be used to send or receive midas events.
        
        Args:
            * buffer_name (str) - The name of the buffer to open. "SYSTEM" is
                the main midas buffer, but other clients can read/write to/from
                their own buffers if desired.
            * buf_size (int) - The size of the buffer to open, in bytes. If not
                specified, defaults to 10 * max_event_size.
            * max_event_size (int) - The size of the largest event we expect
                to receive/send. Defaults to the value of the ODB setting
                "/Experiment/MAX_EVENT_SIZE".
                
        Returns:
            int, the "buffer handle" that must be passed to other buffer/event-
                related functions.
        """
        if max_event_size is None:
            max_event_size = self.odb_get("/Experiment/MAX_EVENT_SIZE")
            
        if buf_size is None:
            buf_size = max_event_size * 10
        
        c_name = ctypes.create_string_buffer(bytes(buffer_name, "utf-8"))
        buffer_size = ctypes.c_int(buf_size)
        buffer_handle = ctypes.c_int()
        
        # Midas allocates the main buffer for us...
        self.lib.c_bm_open_buffer(c_name, buffer_size, ctypes.byref(buffer_handle))
        
        # But we need to allocate space we'll extract each event into.
        # In future we might want to allow using a ring buffer as well as
        # this single-event buffer.
        self.event_buffers[buffer_handle.value] = ctypes.create_string_buffer(max_event_size + 100)
        return buffer_handle.value

    def send_event(self, buffer_handle, event):
        """
        Send an event into a midas buffer.
        
        Args:
            * buffer_handle (int) - The return value from a previous call to
                `open_event_buffer()`.
            * event (`midas.event.Event`) - The event to write.
            
        Example:
        
        ```
        import midas.client
        import midas.event
        
        client = midas.client.MidasClient("MyClientName")
        buffer_handle = client.open_event_buffer("SYSTEM")
        
        event = midas.event.Event()
        event.header.event_id = 123
        event.header.serial_number = 1
        event.create_bank("MYBK", midas.TID_WORD, [1,2,3,4])
        
        client.send_event(buffer_handle, event)
        ```
        """
        buf = event.pack()
        rpc_mode = 1
        
        self.lib.c_rpc_send_event(buffer_handle, buf, len(buf), midas.BM_WAIT, rpc_mode)
        self.lib.c_rpc_flush_event()
        self.lib.c_bm_flush_cache(buffer_handle, midas.BM_WAIT)
        
    def register_event_request(self, buffer_handle, event_id=-1, trigger_mask=-1, sampling_type=midas.GET_ALL):
        """
        Register to be told about events matching certain criteria in a given
        buffer.
        
        To actually get the events, you must later call `receive_event()`.
        
        See `examples/event_receiver.py` for a full example.
        
        Args:
            * buffer_handle (int) - The return value from a previous call to
                `open_event_buffer()`.
            * event_id (int) - Limit to only events where the event ID matches
                this. -1 means no filtering.
            * trigger_mask (int) - Limit to only events where the trigger mask matches
                this. -1 means no filtering.
            * sampling_type (int) - One of :
                `midas.GET_ALL` (get all events)
                `midas.GET_NONBLOCKING` (get as many as possible without blocking producer)
                `midas.GET_RECENT` (like non-blocking, but only get events < 1s old)
                
        Returns:
            int, the "request ID", which can later be used to cancel this
                event request.
        """
        request_id = ctypes.c_int()
        c_event_id = ctypes.c_short(event_id)
        c_trigger_mask = ctypes.c_short(trigger_mask)
        c_sampling_type = ctypes.c_short(sampling_type)
        
        self.lib.c_bm_request_event(buffer_handle, c_event_id, c_trigger_mask, c_sampling_type, ctypes.byref(request_id))
        
        return request_id.value
    
    def receive_event(self, buffer_handle, async_flag=True, use_numpy=False):
        """
        Receive an event from a buffer. You must previously have called
        `open_event_buffer()` to open the buffer and `register_event_request()`
        to define which events you want to be told about.
        
        See `examples/event_receiver.py` for a full example.
        
        Args:
            * buffer_handle (int) - The return value from a previous call to
                `open_event_buffer()`.
            * async_flag (bool) - If True, we'll return None if there is not
                currently an event in the buffer. If False, we'll wait until
                an event is ready and then return.
            * use_numpy (bool) - Whether to use numpy arrays or regular python 
                tuples for bank data in the received `Event`.
            
        Returns:
            `midas.event.Event`
        
        """
        # Buffer for unpacking event
        buf = self.event_buffers[buffer_handle]
        buf_size = ctypes.c_int(ctypes.sizeof(buf))
        c_async_flag = ctypes.c_int(async_flag)
        
        try:
            self.lib.c_bm_receive_event(buffer_handle, ctypes.byref(buf), ctypes.byref(buf_size), c_async_flag)
        except midas.MidasError as e:
            if e.code == midas.status_codes["BM_ASYNC_RETURN"]:
                return None
            else:
                raise
            
        # buf now contains the event contents
        event = midas.event.Event()
        event.unpack(buf, use_numpy=use_numpy)
        
        return event
    
    def deregister_event_request(self, buffer_handle, request_id):
        """
        Cancel an event request that had previously been opened with 
        `register_event_request()`.
        
        Args:
            * buffer_handle (int) - The return value from a previous call to
                `open_event_buffer()`.
            * request_id (int) - The return value from a previous call to
                `register_event_request()`.
        """
        self.lib.c_bm_remove_event_request(buffer_handle, request_id)        

    def start_run(self, run_num=0, async_flag=False):
        """
        Start a new run.
        
        Args:
            * run_num (int) - The new run number. If 0, the run number will
                be automatically incremented.
            * async_flag (bool) - If True, this function will return as soon
                as the transition has started; if False, it will wait until
                the transition has completed (or failed). 
                
        Raises:
            `midas.TransitionFailedError` if async_flag is False and the 
                transition fails.
        """
        self._run_transition(midas.TR_START, run_num, async_flag)
        
    def stop_run(self, async_flag=False):
        """
        Stop a run.
        
        Args:
            * async_flag (bool) - If True, this function will return as soon
                as the transition has started; if False, it will wait until
                the transition has completed (or failed). 
                
        Raises:
            `midas.TransitionFailedError` if async_flag is False and the 
                transition fails.
        """        
        self._run_transition(midas.TR_STOP, 0, async_flag)
    
    def pause_run(self, async_flag=False):
        """
        Pause a run.
        
        Args:
            * async_flag (bool) - If True, this function will return as soon
                as the transition has started; if False, it will wait until
                the transition has completed (or failed). 
                
        Raises:
            `midas.TransitionFailedError` if async_flag is False and the 
                transition fails.
        """        
        self._run_transition(midas.TR_PAUSE, 0, async_flag)
        
    def resume_run(self, async_flag=False):
        """
        Resume a paused run.
        
        Args:
            * async_flag (bool) - If True, this function will return as soon
                as the transition has started; if False, it will wait until
                the transition has completed (or failed). 
                
        Raises:
            `midas.TransitionFailedError` if async_flag is False and the 
                transition fails.
        """        
        self._run_transition(midas.TR_RESUME, 0, async_flag)
        
    def register_transition_callback(self, transition, sequence, callback):
        """
        Register a function to be called when a transition happens.
        
        Args:
            * transition (int) - One of `midas.TR_START`, `midas.TR_STOP`,
                `midas.TR_PAUSE`, `midas.TR_RESUME`, `midas.TR_STARTABORT`.
            * sequence (int) - The order in which this function will be 
                called.
            * callback (function) - See below.
            
        Python function (`callback`) details:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
                * run_number (int) - The current/new run number.
            * Value it should return - choose from:
                * None or 1 - Transition was successful
                * int that isn't 1 - Transition failed
                * 2-tuple of (int, str) - Transition failed and the reason why.
        """
        cb = midas.callbacks.make_transition_callback(callback, self)
        self.lib.c_cm_register_transition(transition, cb, sequence)

    def set_transition_sequence(self, transition, sequence):
        """
        Update the order in which our transition callback functions are called.
        
        Args:
            * transition (int) - One of `midas.TR_START`, `midas.TR_STOP`,
                `midas.TR_PAUSE`, `midas.TR_RESUME`, `midas.TR_STARTABORT`.
            * sequence (int) - The order in which this function will be 
                called.
        """
        self.lib.c_cm_set_transition_sequence(transition, sequence)
    
    def deregister_transition_callback(self, transition):
        """
        Deregister functions previously registered with 
        `register_transition_callback()`, so they are no longer called.
        
        Args:
            * transition (int) - One of `midas.TR_START`, `midas.TR_STOP`,
                `midas.TR_PAUSE`, `midas.TR_RESUME`, `midas.TR_STARTABORT`.
        """
        self.lib.c_cm_deregister_transition(transition)
    
    def register_deferred_transition_callback(self, transition, callback):
        """
        Register a function that can defer a run transition. No client will 
        receive the actual transition until your callback function says that 
        it is okay.
        
        Args:
            * transition (int) - One of `midas.TR_START`, `midas.TR_STOP`,
                `midas.TR_PAUSE`, `midas.TR_RESUME`, `midas.TR_STARTABORT`.
            * callback (function) - See below. This will be called frequently
                by midas during the deferral period, and tells midas whether
                the transition can proceed or not yet.
            
        Python function (`callback`) details:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
                * run_number (int) - The current/new run number
            * Value it should return:
                bool - True if the transition can proceed; False if the 
                    transition should wait.
        
        """
        cb = midas.callbacks.make_deferred_transition_callback(callback, self)
        self.lib.c_cm_register_deferred_transition(transition, cb)

    def register_disconnect_callback(self, callback):
        """
        Register a function that will be called immediately before we disconnect
        from midas.

        To GUARANTEE that this function gets called, you should use a context
        manager style for the midas client (`with` syntax).

        ```
        # This style will work in most cases, but is not guaranteed to call
        # the callback when the program exits (depends on your specific 
        # python version/implementation and python code):
        client = midas.client.MidasClient("pytest")
        client.register_disconnect_callback(some_func)
        exit(2)

        # This style is guaranteed to work:
        with midas.client.MidasClient("pytest") as client:
            client.register_disconnect_callback(some_func)
            exit(2)
        ```

        Args:
            * callback (function) - See below.
        
        Python function (`callback`) details:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
            * Value it should return:
                Anything or nothing.
        """
        midas.callbacks.make_and_store_disconnect_callback(callback, self)

    def deregister_disconnect_callback(self):
        """
        De-register a function that was previously passed to
        `register_disconnect_callback()`.
        """
        midas.callbacks.make_and_store_disconnect_callback(None, self)

    def get_midas_version(self):
        """
        Get the version of midas.
        
        Returns:
            2-tuple of (str, str) for (midas version, git revision)
        """
        ver = self.lib.c_cm_get_version()
        rev = self.lib.c_cm_get_revision()
        
        return (ver.decode("utf-8"), rev.decode("utf-8"))
    
    def get_experiment_dir(self):
        """
        Get the "experiment directory" on disk. Same as cm_get_path() in C++ code.

        Returns:
            str
        """
        max_len = 256
        c_path = ctypes.create_string_buffer(max_len)
        self.lib.c_cm_get_path(c_path, max_len)
        return c_path.value.decode('utf-8')
    
    def start_client(self, client_name, check_if_already_exists_before_starting=False):
        """
        Start a client/program that has been defined in /Programs in the ODB.

        Must have the ODB key `/Programs/<client_name>/Start command` set, as this is the
        command we'll execute to try to start the client.

        You can later use `client_exists()` to check whether the client started 
        successfully (e.g. wait a few seconds and then call `client_exists()`).

        Args:
            * client_name (str) - The name of a client/program defined in /Programs in the ODB.
            * check_if_already_exists_before_starting (bool) - If True, we won't do anything
                if midas thinks the client/program is already running.

        Returns:
            int, the PID of the spawned process. Most users won't care about this.

        Raises:
            * KeyError if the client/program isn't found in /Programs.
            * MidasError if `fork()` failed or similar.
        """
        if check_if_already_exists_before_starting and self.client_exists(client_name):
            # No need to start program if already running
            return
        
        cmd = self.odb_get(f"/Programs/{client_name}/Start command")

        return self.exec(cmd)

    def stop_client(self, client_name, exact_match=True):
        """
        Stop a client/program that is running (and is connected to midas).

        Args:
            * client_name (str) - The name of a midas client that is connected.
                You can pass the special value "all" to shutdown all clients.
            * exact_match (bool) - If False, we'll look for all clients that
                start with this name. E.g. "myfe" would match "myfe_1", "myfetest"
                etc. If True, we require an exact match of the name.
        """
        c_name = ctypes.create_string_buffer(bytes(client_name, "utf-8"))
        c_unique = ctypes.c_uint32(exact_match)
        self.lib.c_cm_shutdown(c_name, c_unique)

    def get_all_required_program_names(self):
        """
        Get all the clients/programs listed in /Programs in the ODB that have the
        "Required" key set to True.

        Returns:
            list of str, names that can be passed to `start_client()`, `stop_client`,
            `connect_to_other_client()` etc.
        """
        return [k for k,v in self.odb_get("/Programs").items() if v["Required"]]

    def start_all_required_programs(self):
        """
        Start all the programs listed in /Programs in the ODB that have the "Required"
        key set to True and that are not already running.
        """
        client_names = self.get_all_required_program_names()
        for name in client_names:
            self.start_client(name, True)
    
    def client_exists(self, client_name, exact_match=True):
        """
        Check whether a client/program is running.

        Args:
            * client_name (str) - name of the other client
            * exact_match (bool) - If False, we'll look for all clients that
                start with this name. E.g. "myfe" would match "myfe_1", "myfetest"
                etc. If True, we require an exact match of the name.

        Returns:
            bool
        """
        c_name = ctypes.create_string_buffer(bytes(client_name, "utf-8"))
        c_unique = ctypes.c_uint32(exact_match)
        status = self.lib.c_cm_exist(c_name, c_unique)
        return status == midas.status_codes["SUCCESS"]

    def clients_exist(self, client_names, return_as_dict=False):
        """
        Check whether multiple clients/programs are running.

        Args:
            * client_names (list of str) - names of the clients to check.
            * return_as_dict (bool) - whether to return a single boolean
                or a dict with the per-client status.

        Returns:
            If return_as_dict is True: a dict of {str: bool} for whether each
                client is running or not.
            If return_as_dict is False: a boolean of whether all the clients/
                programs are running or not. If any client/program is not running,
                we'll return False.
        """
        if not isinstance(client_names, list):
            client_names = [client_names]

        ret_dict = {}
        ret_bool = True

        for name in client_names:
            this_exists = self.client_exists(name)
            ret_dict[name] = this_exists

            if not this_exists:
                ret_bool = False

        return ret_dict if return_as_dict else ret_bool
    
    def exec(self, command):
        """
        Execute a command in a separate process.

        Uses `fork()` and `execl()` system calls to run the subprocess.

        Args:
            * command (str) - Command to execute, e.g. the path to a frontend executable.

        Returns:
            int, the PID of the spawned process. Most users won't care about this.
        """
        c_child_pid = ctypes.c_int()
        c_command = ctypes.create_string_buffer(bytes(command, "utf-8"))
        self.lib.c_ss_exec(c_command, ctypes.byref(c_child_pid))
        return c_child_pid.value
    
    def connect_to_other_client(self, other_client_name):
        """
        Connect to a different midas client, so that you can later call RPC
        functions exposed by that client.
        
        Args:
            * other_client_name (str)
            
        Returns:
            int, which you should pass to `disconnect_from_other_client` and
            `jrpc_client_call`.
        """
        c_name = ctypes.create_string_buffer(bytes(other_client_name, "utf-8"))
        handle = ctypes.c_int()
        self.lib.c_cm_connect_client(c_name, ctypes.byref(handle))
        return handle
        
    def disconnect_from_other_client(self, connection, shutdown_other_client=False):
        """
        Disconnect from a client that was previously connected to using
        `connect_to_other_client`.
        
        Args:
            * connection (int) - The connection handle returned to you when 
                connecting.
            * shutdown_other_client (bool) - Whether to cause the other 
                client to exit.
        """
        shutdown = ctypes.c_uint32(1 if shutdown_other_client else 0)
        self.lib.c_cm_disconnect_client(connection, shutdown)
        
    def jrpc_client_call(self, connection, cmd, args, max_len=1024):
        """
        Call the JRPC (aka "javascript RPC") function exposed by another client.

        For JRPC, the other client returns a string of data. Use `brpc_client_call()`
        if the other client will be returning arbitrary binary data.
        
        Args:
            * connection (int) - The connection handle returned to you when 
                connecting.
            * cmd (str) - The command to execute. The other client's JRPC 
                handler must know what to do with this.
            * args (str) - Any other arguments to pass to the other client.
                Specify as a single string (probably separating arguments
                with spaces).
            * max_len (int) - The maximum length you expect the returned string
                to be. This is an imposition of the underlying C++ code, not
                the python side.
                
        Returns:
            str, the string populated by the other client. Note that an exception
            will be raised if the other client returns a non-SUCESSS status.
        """
        c_cmd = ctypes.create_string_buffer(bytes(cmd, "utf-8"))
        c_args = ctypes.create_string_buffer(bytes(args, "utf-8"))
        c_buf = ctypes.create_string_buffer(max_len)
        self.lib.c_jrpc_client_call(connection, c_cmd, c_args, c_buf, max_len)
        return c_buf.value.decode('utf-8')
            
    def brpc_client_call(self, connection, cmd, args, as_event=False, use_numpy=False, max_len=1000000):
        """
        Call the BRPC (aka "binary RPC") function exposed by another client.

        For BRPC, the other client returns arbitrary binary data (possibly encoded as
        a midas event). Use `jrpc_client_call()` if the other client will be returning 
        a string.
        
        Args:
            * connection (int) - The connection handle returned to you when 
                connecting.
            * cmd (str) - The command to execute. The other client's BRPC 
                handler must know what to do with this.
            * args (str) - Any other arguments to pass to the other client.
                Specify as a single string (probably separating arguments
                with spaces).
            * as_event (bool) - Whether you expect the data from the other client
                to be arbitrary binary data, or encoded as a midas event. If True,
                we'll automatically decode the event for you.
            * use_numpy (bool) - Is `as_event` is True, whether to use numpy arrays 
                or regular python tuples for bank data in the received `Event`.
            * max_len (int) - The maximum length you expect the returned data
                to be (in bytes). This is an imposition of the underlying C++ code, 
                not the python side.
                
        Returns:
            The data populated by the other client.
              If `as_event` is False: a `ctypes.Array` of `ctypes.char_c`.
              If `as_event` is True: a `midas.event.Event`.
            Note that an exception will be raised if the other client returns a non-SUCESSS status.
        """
        c_cmd = ctypes.create_string_buffer(bytes(cmd, "utf-8"))
        c_args = ctypes.create_string_buffer(bytes(args, "utf-8"))
        c_buf = ctypes.create_string_buffer(max_len)
        c_max_len = ctypes.c_int(max_len)
        self.lib.c_brpc_client_call(connection, c_cmd, c_args, c_buf, ctypes.byref(c_max_len))

        if as_event:
            ev = midas.event.Event()
            ev.unpack(c_buf, 0, use_numpy)
            return ev
        else:
            return c_buf[:c_max_len.value]
        
    def register_jrpc_callback(self, callback, return_success_even_on_failure=False):
        """
        Register a function that can be called from mhttpd, custom webpages, and
        other clients (using the "javascript RPC" system). You should register a 
        single function, but can do multiple things in that function based on the 
        parameters provided by the javascript call.

        For jrpc you return a string, and it will be interpreted as a string by
        the calling function. See `register_brpc_callback()` (brpc not jrpc) for
        a version where you can return arbitrary binary data (potentially encoded
        as a midas event).
        
        Args:
            * callback (function) - See below.
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
        cb = midas.callbacks.make_rpc_callback(callback, self, return_success_even_on_failure)
        self.lib.c_cm_register_function(midas.RPC_JRPC, cb)

    def register_brpc_callback(self, callback):
        """
        Register a function that can be called from mhttpd, custom webpages, and
        other clients (using the "javascript RPC" system). You should register a 
        single function, but can do multiple things in that function based on the 
        parameters provided by the javascript call.

        For brpc you return arbitrary binary data. See `register_jrpc_callback()` 
        (jrpc not brpc) for a version where you jsut want to return a string.
        
        Args:
            * callback (function) - See below.
            
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
        cb = midas.callbacks.make_brpc_callback(callback, self)
        self.lib.c_cm_register_function(midas.RPC_BRPC, cb)

    def trigger_internal_alarm(self, alarm_name, message, default_alarm_class="Alarm"):
        """
        Trigger an "internal" alarm (i.e. an alarm raised directly by code, rather than due
        to an ODB condition failing or a program not running).

        Example code:

        ```
        client.trigger_internal_alarm("Example", "Oh no, something went wrong!")
        client.get_triggered_alarms()
        # Returns {'Example': 'Oh no, something went wrong!'}
        client.reset_alarm("Example")
        client.get_triggered_alarms()
        # Returns {}
        ```

        Args:
            * alarm_name (str) - The name of this alarm. An entry will be created in the
                ODB at `/Alarms/Alarms/<alarm_name>` if it doesn't already exist. Max 32
                characters, and not "/" characters allowed.
            * message (str) - The message to be displayed to users. Max 80 characters.
            * default_alarm_class (str) - The "class" of this alarm. (i.e. what to do when 
                this alarm is triggered). See `create_alarm_class()` for more about alarm
                classes. If an alarm already exists called `alarm_name`, the class of that
                alarm will not be changed.
        """
        c_name = ctypes.create_string_buffer(bytes(alarm_name, "utf-8"))
        c_msg = ctypes.create_string_buffer(bytes(message, "utf-8"))
        c_cond = ctypes.create_string_buffer(bytes(message, "utf-8"))
        c_class = ctypes.create_string_buffer(bytes(default_alarm_class, "utf-8"))
        c_type = ctypes.c_int(midas.AT_INTERNAL)
        self.lib.c_al_trigger_alarm(c_name, c_msg, c_class, c_cond, c_type)

    def create_evaluated_alarm(self, alarm_name, odb_condition, message=None, alarm_class="Alarm", activate_immediately=True):
        """
        Create an alarm that will trigger if a condition in the ODB is met.

        To deactivate this alarm, set `/Alarms/Alarms/<alarm_name>/Active` to False in the ODB.
        To completely delete this alarm, just delete `/Alarms/Alarms/<alarm_name>` from the ODB.

        Args:
            * alarm_name (str) - The name of this alarm. An entry will be created in the
                ODB at `/Alarms/Alarms/<alarm_name>` if it doesn't already exist. Max 32
                characters, and not "/" characters allowed.
            * odb_condition (str) - What to check for. Examples are "/Path/to/float < 0.3"
                and "/Path/to/boolean == y". Max 256 characters.
            * message (str) - The message to be displayed to users. Max 80 characters. If not
                set, the message will be the same as `odb_condition`.
            * alarm_class (str) - The "class" of this alarm (i.e. what to do when this alarm is
                triggered). See `create_alarm_class()` for more about alarm classes.
            * activate_immediately (bool) - Whether to activate this alarm immediately or not.
                (This doesn't mean it will actually trigger, just that midas will start evaluating
                whether to trigger - "active" and "triggered" have different meanings).
        """
        c_name = ctypes.create_string_buffer(bytes(alarm_name, "utf-8"))
        c_cond = ctypes.create_string_buffer(bytes(odb_condition, "utf-8"))

        if message is None:
            message = odb_condition

        c_msg = ctypes.create_string_buffer(bytes(message, "utf-8"))
        c_class = ctypes.create_string_buffer(bytes(alarm_class, "utf-8"))
        self.lib.c_al_define_odb_alarm(c_name, c_cond, c_class, c_msg)

        self.odb_set("/Alarms/Alarms/%s/Active" % alarm_name, activate_immediately)

    def create_alarm_class(self, class_name, execute_command="", execute_interval_secs=0, stop_run=False):
        """
        Alarm classes define what to do when an alarm is triggered. All alarms belong
        to a single class. You can create as many classes as you like. The alarm classes
        "Alarm" and "Warning" come pre-defined by midas.

        Args:
            * class_name (str) - The name of this alarm class. An entry will be created 
                in the ODB at `/Alarms/Classes/<class_name>` if it doesn't already exist
            * execute_command (str) - System command to run when an alarm with this class
                is triggered (e.g. "python /path/to/some/script.py" or "~/path/to/script.sh")
            * execute_interval_secs (int) - How often to run the above command, in seconds. 
                NOTE - if this is 0, THE COMMAND WILL NOT BE RUN AT ALL! If you just want the
                command to be run once, when the alarm is first raised, set a very high value
                like 1e9 (but < 2^31 as the value is a signed 32-bit int in the ODB).
            * stop_run (bool) - Whether to stop the run when an alarm of this class is triggered.
        """
        odb_dict = collections.OrderedDict([("Write system message", True),
                                            ("Write Elog message", False),
                                            ("System message interval", 60),
                                            ("System message last", ctypes.c_uint32(0)),
                                            ("Execute command", ctypes.create_string_buffer(bytes(execute_command, "utf-8"), 256)),
                                            ("Execute interval", ctypes.c_int32(execute_interval_secs)),
                                            ("Execute last", ctypes.c_uint32(0)),
                                            ("Stop run", bool(stop_run)),
                                            ("Display BGColor", ctypes.create_string_buffer(b"red", 32)),
                                            ("Display FGColor", ctypes.create_string_buffer(b"black", 32)),
                                            ("Alarm sound", True)])
        self.odb_set("/Alarms/Classes/%s" % class_name, odb_dict)

    def get_triggered_alarms(self):
        """
        Get all the alarms that have been triggered but not yet reset.

        Returns:
            dict of {str: str}. Key is the alarm name, value is the alarm message.
        """
        alarms = self.odb_get("/Alarms/Alarms")
        system_active = self.odb_get("/Alarms/Alarm System active")

        retval = {}

        if system_active:
            for alarm_name, state in alarms.items():
                if state["Active"] and state["Triggered"]:
                    retval[alarm_name] = state["Alarm Message"]

        return retval

    def reset_alarm(self, alarm_name):
        """
        Reset an alarm that is currently triggered.

        Note that if it as an evaluated alarm, midas may re-trigger the alarm
        again very quickly if the ODB condition is still met.

        Args:
            * alarm_name (str) - The name of the alarm to reset.
        """
        c_name = ctypes.create_string_buffer(bytes(alarm_name, "utf-8"))
        self.lib.c_al_reset_alarm(c_name)

    def get_message_facilities(self):
        """
        Get a list of "facilties" that send messages to the message log.
        By default the "midas" facility is used when sending messages, but
        others can be used. The facility name can be used when calling
        `get_recent_messages()`.

        Returns:
            list of string
        """
        arr = ctypes.POINTER(ctypes.c_char_p)()
        arr_len = ctypes.c_int()
        self.lib.c_cm_msg_facilities(ctypes.byref(arr), ctypes.byref(arr_len))
        casted = ctypes.cast(arr, ctypes.POINTER(ctypes.c_char_p))
        py_list = [casted[i].decode("utf-8") for i in range(arr_len.value)]
        self.lib.c_free_list(arr, arr_len)
        return py_list

    def get_recent_messages(self, min_messages=1, before=None, facility="midas"):
        """
        Get a list of messages that were sent to the midas message log before a
        certain time.

        The number of messages returned is based on:
            * Search until we find `min_messages`
            * If there are more messages with the same UNIX timestamp (to 1s
                resolution), include those messages as well.
            * If log files are being split by date, and we've had to look back
                through several files without finding anything, give up.

        Args:
            * min_messages (int) - Minimum number of messages to try to retrieve.
            * before (datetime.datetime, number or None) - Start searching for messages
                before the specified time. None means get the most recent. A number is
                interpreted as a UNIX timestamp.
            * facility (str) - The "facility" to search in. By default most messages
                are written to the "midas" facility.

        Returns:
            list of string - messages, with the most recent one first
        """
        c_facility = ctypes.create_string_buffer(bytes(facility, "utf-8"))
        c_n_msg = ctypes.c_int(min_messages)
        c_msgs = ctypes.c_char_p()
        c_num_msgs_recvd = ctypes.c_int()

        # Convert timestamp to a number
        if before is None:
            before = 0

        if isinstance(before, datetime.datetime):
            before = before.timestamp()

        c_before = ctypes.c_uint64(int(before))
        
        # Get messages
        self.lib.c_cm_msg_retrieve2(c_facility, c_before, c_n_msg, ctypes.byref(c_msgs), ctypes.byref(c_num_msgs_recvd))
        
        # Convert \n-separated value to list of python strings
        pylist = c_msgs.value.decode('utf-8').split("\n")
        self.lib.c_free(c_msgs)

        return [m for m in pylist if len(m) > 0]

    def register_message_callback(self, callback, only_message_types=[midas.MT_INFO, midas.MT_ERROR]):
        """
        Register a function to be called when a message is sent to the message log.
        Note you will receive messages from all "facilities" - it is not possible
        to only request messages from a single facility.
        
        Args:
            * transition (int) - One of `midas.TR_START`, `midas.TR_STOP`,
                `midas.TR_PAUSE`, `midas.TR_RESUME`, `midas.TR_STARTABORT`.
            * sequence (int) - The order in which this function will be 
                called.
            * callback (function) - See below.
            * only_message_types (None, or list of midas.MT_xxx flags) - which message types
                to pass to the callback function (e.g. only midas.MT_ERROR, not midas.MT_INFO)
            
        Python function (`callback`) details:
            * Arguments it should accept:
                * client (midas.client.MidasClient)
                * msg (str) - the actual message
                * timestamp (datetime.datetime) - the timestamp of the message
                * msg_type (int) - midas.MT_xxx flag, e.g. midas.MT_ERROR
            * Value it should return:
                * Anything or nothing, we don't do anything with it 
        """
        if only_message_types is not None and not self._is_list_like(only_message_types):
            only_message_types = [only_message_types]
        cb = midas.callbacks.make_msg_handler_callback(callback, self, only_message_types)
        self.lib.c_cm_msg_register(cb)

    def deregister_message_callback(self):
        """
        Unregister a callback that was registered with `register_message_callback()`.
        """
        # No clean way to do this with C++ API, but closing and re-opening the
        # message buffer does the trick.
        self.lib.c_cm_msg_close_buffer()
        self.lib.c_cm_msg_open_buffer()

    def hist_get_events(self):
        """
        Get a list of valid history "events". Use the event name when calling
        `hist_get_tags()` or `hist_get_data()`.

        Returns:
            list of string
        """
        arr = ctypes.POINTER(ctypes.c_char_p)()
        arr_len = ctypes.c_int()

        self.lib.c_hs_get_events(self.hDB, ctypes.byref(arr), ctypes.byref(arr_len))

        casted = ctypes.cast(arr, ctypes.POINTER(ctypes.c_char_p))
        py_list = [casted[i].decode("utf-8") for i in range(arr_len.value)]
        self.lib.c_free_list(arr, arr_len)

        return py_list
    
    def hist_get_tags(self, event_name):
        """
        Get a list of valid history "tags" for a given history "event".
        Find valid event names using `hist_get_events()`. Use event name
        and tag name when calling `hist_get_data()`.

        Returns:
            list of dicts like:
                [{"name": string, "type": int, "n_data": int}]
            where 
            * "name" is the tag name that can be passed to 
            * "type" is the midas type (e.g. TID_DOUBLE = 10), 
            * "n_data" is the number of entries if it's an array.
        """
        # Throw early if not a valid event name
        self._hist_validate_event_name(event_name)

        # Event name is valid
        c_event_name = ctypes.create_string_buffer(bytes(event_name, "utf-8"))
        arr_names = ctypes.POINTER(ctypes.c_char_p)()
        arr_types = ctypes.c_void_p()
        arr_n_data = ctypes.c_void_p()
        arr_len = ctypes.c_int()

        self.lib.c_hs_get_tags(self.hDB, c_event_name, ctypes.byref(arr_names), ctypes.byref(arr_types), ctypes.byref(arr_n_data), ctypes.byref(arr_len))
        
        casted_names = ctypes.cast(arr_names, ctypes.POINTER(ctypes.c_char_p))
        casted_types = ctypes.cast(arr_types, ctypes.POINTER(ctypes.c_uint32))
        casted_n_data = ctypes.cast(arr_n_data, ctypes.POINTER(ctypes.c_uint32))

        py_names = [casted_names[i].decode("utf-8") for i in range(arr_len.value)]
        py_types = casted_types[:arr_len.value]
        py_n_data = casted_n_data[:arr_len.value]

        self.lib.c_free_list(arr_names, arr_len)
        self.lib.c_free(arr_types)
        self.lib.c_free(arr_n_data)

        retval = []
        
        for i in range(len(py_names)):
            retval.append({"name": py_names[i], "type": py_types[i], "n_data": py_n_data[i]})

        return retval

    def hist_get_recent_data(self, num_hours, interval_secs, event_name, tag_name, index=None, timestamps_as_datetime=False):
        """
        Convenience wrapper around `hist_get_data()` for getting the most recent
        N hours of data from the history system.

        Args:
            * num_hours (int/float) - The number of hours of data to retrieve
            * All others - see `hist_get_data()`

        Returns:
            See `hist_get_data()`
        """
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=num_hours)
        return self.hist_get_data(start, end, interval_secs, event_name, tag_name, index, timestamps_as_datetime)

    def hist_get_data(self, start_time, end_time, interval_secs, event_name, tag_name, index=None, timestamps_as_datetime=False):
        """
        Read data from the history system.

        Args:
            * start_time (int/float/datetime.datetime) - Start of the time period 
                to retrieve data for. If a number, it's interpreted as a UNIX timestamp.
            * end_time (int/float/datetime.datetime) - End of the time period 
                to retrieve data for. If a number, it's interpreted as a UNIX timestamp.
            * interval_secs (int) - Return at most one data point per N seconds.
            * event_name (str) - Name of the midas history "event" to query. Use
                `hs_get_events()` to find valid event names.
            * tag_name (str) - Name of the midas history "tag" to query. Use
                `hs_get_tags()` to find valid tag names for a given event.
            * index (int/None) - If the tag represents an array, you can ask to get data
                for a specific element in that array. If None, you'll get data for all
                the elements in the array. The length of the array is returned as part
                of `hs_get_tags()`.
            * timestamps_as_datetime (bool) - Whether to convert timestamps in the
                return value to `datetime.datetime` objects, or leave them as integer 
                UNIX timestamps (the default).
        
        Returns:
            A list of dicts. One dict per event/tag/index combination. The dict
            has the following keys:
            {
                "event_name": str, 
                "tag_name": str, 
                "index": int, 
                "num_entries": int,
                "status": int,
                "timestamps": list of int/datetime.datetime,
                "values": list of float
            }

            * "index" is the element index if a tag represents an array of data.
            * "num_entries" is how many timestamp/value pairs were found.
            * "status" is a midas status code for this event/tag/index combination.
                1 indicates SUCCESS. 702 indicates data we couldn't find any data.
                Any other status codes will result in an exception being thrown.
            * "timestamps" is the timestamp of each data point found.
            * "values" is the value of each data point found.

            The `timestamps` and `values` arrays will be the same length.
        """
        # Throw early if not a valid event/tag name combination. Do this in
        # python as some paths in C++ return SUCCESS even if event/tag not found.
        self._hist_validate_event_name(event_name)
        tags = self.hist_get_tags(event_name)
        self._hist_validate_tag_name(tag_name, tags)

        # Is a valid event/tag name combination
        if isinstance(start_time, datetime.datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime.datetime):
            end_time = int(end_time.timestamp())
        
        c_start_time = ctypes.c_uint32(start_time)
        c_end_time = ctypes.c_uint32(end_time)
        c_interval = ctypes.c_uint32(interval_secs)
        c_event_name = ctypes.create_string_buffer(bytes(event_name, "utf-8"))
        c_tag_name = ctypes.create_string_buffer(bytes(tag_name, "utf-8"))

        # Default to reading a single variable, unless
        # user wants us to read all variables in an array
        n_vars = 1

        if index is None:
            index = 0

            for tag in tags:
                if tag["name"] == tag_name:
                    n_vars = tag["n_data"]
        elif not isinstance(index, int):
            raise ValueError("variable index must be an integer or None")
        
        c_index = ctypes.c_int(index)
        c_n_vars = ctypes.c_int(n_vars)

        arr_num_entries = ctypes.c_void_p()
        arr_times = ctypes.c_void_p()
        arr_values = ctypes.c_void_p()
        arr_hs_status = ctypes.c_void_p()

        # arr_num_entries and arr_hs_status are len n_vars
        # arr_times and arr_values are len of sum(arr_num_entries)
        self.lib.c_hs_read(self.hDB, c_start_time, c_end_time, c_interval, 
                           c_event_name, c_tag_name, c_index, c_n_vars, 
                           ctypes.byref(arr_num_entries),
                           ctypes.byref(arr_times),
                           ctypes.byref(arr_values),
                           ctypes.byref(arr_hs_status))
                
        casted_num_entries = ctypes.cast(arr_num_entries, ctypes.POINTER(ctypes.c_int32))
        casted_times = ctypes.cast(arr_times, ctypes.POINTER(ctypes.c_uint32))
        casted_values = ctypes.cast(arr_values, ctypes.POINTER(ctypes.c_double))
        casted_hs_status = ctypes.cast(arr_hs_status, ctypes.POINTER(ctypes.c_int32))

        py_num_entries = casted_num_entries[:n_vars]
        py_hs_status = casted_hs_status[:n_vars]

        retval = []
        buf_offset = 0
        raise_status = None
        
        for i in range(n_vars):
            n = py_num_entries[i]
            status = py_hs_status[i]

            if status != midas.status_codes["SUCCESS"] and status != midas.status_codes["HS_FILE_ERROR"]:
                # Ignore status code that means "we couldn't find any data"
                raise_status = status

            obj = {
                "event_name": event_name, 
                "tag_name": tag_name, 
                "index": index + i, 
                "num_entries": n,
                "status": status,
                "timestamps": [],
                "values": []
            }

            if n > 0:
                time_int = casted_times[buf_offset:buf_offset+n]

                if timestamps_as_datetime:
                    time_dt = [datetime.datetime.fromtimestamp(t) for t in time_int]
                    obj["timestamps"] = time_dt
                else:
                    obj["timestamps"] = time_int

                obj["values"] = casted_values[buf_offset:buf_offset+n]

            retval.append(obj)
            buf_offset += n

        self.lib.c_free(arr_num_entries)
        self.lib.c_free(arr_hs_status)
        self.lib.c_free(arr_times)
        self.lib.c_free(arr_values)

        # Only throw after we've freed all the memory that was allocated for us!
        # `status_code_to_exception()` does the raise for us.
        if raise_status:
            midas.status_code_to_exception(raise_status, [])

        return retval
    
    def hist_create_plot(self, group_name, panel_name, variables, labels=[]):
        """
        Helper function for adding a plot to the midas history display. It requires
        you to know how midas refers to different variables within the history
        system, so is an *expert-level function* (or create one plot manually, see
        the pattern of variable names used in the ODB for that plot, then infer
        what the names would be for future plots).

        Arguments:
            * group_name (str) - The group this plot belongs to (used to organise plots
                on the main "History" webpage).
            * panel_name (str) - The name of this plot (called a panel name as you can 
                view all the plots in a group on a single page with multiple panels).
            * variable (list of str) - The variables to show, as referred to by the
                internal midas history system. Combines both the "event name" and "tag name"
                into a single string. E.g. "Lakeshore:L336[0]" or "MyEvent:Custom tag name".
            * labels (list of str) - Human readable name for each variable (optional).

        Raises:
            * `KeyError` if a plot already exists with this name
            * `ValueError` or `TypeError` if there is a problem with the other arguments
        """
        odb_path = "/History/Display/%s/%s" % (group_name, panel_name)

        if self.odb_exists(odb_path):
            raise KeyError("History plot already exists. Not touching it.")
        
        if not isinstance(variables, list):
            raise TypeError("variables argument must be a list")
        
        if not isinstance(labels, list):
            raise TypeError("labels argument must be a list")
        
        if len(labels) > len(variables):
            raise ValueError("Labels list must be same length (or shorter) than variables list")
        
        num_vars = len(variables)

        while len(labels) < num_vars:
            labels.append("")

        default_colours = [ b"#00AAFF", b"#FF9000", b"#FF00A0", b"#00C030",
                            b"#A0C0D0", b"#D0A060", b"#C04010", b"#807060",
                            b"#F0C000", b"#2090A0", b"#D040D0", b"#90B000",
                            b"#B0B040", b"#B0B0FF", b"#FFA0A0", b"#A0FFA0"
                        ]

        c_variables = [ctypes.create_string_buffer(bytes(v, "utf-8"), 64) for v in variables]
        c_labels = [ctypes.create_string_buffer(bytes(l, "utf-8"), 32) for l in labels]
        c_colours = [ctypes.create_string_buffer(default_colours[i%len(default_colours)], 32) for i in range(num_vars)]
        c_formula = [ctypes.create_string_buffer(b"", 64) for i in range(num_vars)]
        
        odb_dict = collections.OrderedDict([("Variables", c_variables),
                                            ("Timescale", ctypes.create_string_buffer(b"1h", 32)),
                                            ("Show values", True),
                                            ("Show fill", True),
                                            ("Show run markers", True),
                                            ("Label", c_labels),
                                            ("Colour", c_colours),
                                            ("Formula", c_formula)])

        self.odb_set(odb_path, odb_dict)

    #
    # Most users should not need to use the functions below here, as they are
    # quite low-level and helper functions for the more user-friendly interface
    # above.
    #
    
    def _rpc_is_remote(self):
        """
        Whether we're connected to a remote mserver rather than running
        locally. Needed so we can avoid calling "RPC-unsafe" functions.
        
        Returns
            bool
        """
        retval = self.lib.c_rpc_is_remote()
        return retval == 1
    
    def _convert_dwords_from_odb(self, curr_place):
        """
        Unsigned ints are returned as JSON strings (like "0x0000").
        Change them to actual numbers. Negative signed bytes may
        alse be returned incorrectly.
        
        Args:
            * curr_place (dict or collections.OrderedDict) - ODB JSON dump
            
        Returns
            dict, with all WORD/DWORD/QWORD/INT64s as actual numbers.
        """
        if not isinstance(curr_place, dict):
            return curr_place
    
        # Initialize as an ordered dict, dict or whatever
        retval = type(curr_place)()
    
        for k,v in curr_place.items():
            if k.endswith("/key"):
                retval[k] = v
                continue
            
            if isinstance(v, dict):
                retval[k] = self._convert_dwords_from_odb(v)
            else:
                meta = curr_place[k+"/key"]
                
                if meta["type"] in (midas.TID_WORD, midas.TID_DWORD, midas.TID_QWORD):
                    if isinstance(v, list):
                        retval[k] = [int(x, 16) for x in v]
                    else:
                        retval[k] = int(v, 16)
                elif meta["type"] == midas.TID_INT64:
                    if isinstance(v, list):
                        retval[k] = [int(x) for x in v]
                    else:
                        retval[k] = int(v)
                elif meta["type"] == midas.TID_SBYTE:
                    if isinstance(v, list):
                        retval[k] = [x-256 if x > 127 else x for x in v]
                    else:
                        retval[k] = v-256 if v > 127 else v
                else:
                    retval[k] = v
    
        return retval
    
    def _prune_metadata_from_odb(self, curr_place):
        """
        If an ODB query has included "/key" entries, you can strip that
        metadata out with this function. We recurse down the whole ODB
        structure in necessary.
    
        Args:
    
        * curr_place (dict or collections.OrderedDict) - ODB structure
    
        Returns:
            dict, with all "XXX/key" entries removed.
        """
        if not isinstance(curr_place, dict):
            return curr_place
    
        # Initialize as an ordered dict, dict or whatever
        retval = type(curr_place)()
    
        for k,v in curr_place.items():
            if k.endswith("/key"):
                continue
            if isinstance(v, dict):
                retval[k] = self._prune_metadata_from_odb(v)
            else:
                retval[k] = v
    
        return retval
    
    def _odb_path_is_for_single_array_element(self, path):
        """
        Whether a user has specified an ODB path that references a single
        element in an array.
        
        Args:
            * path (str)
            
        Returns:
            3-tuple of (bool, int/None, ctypes string buffer) for
                Whether it's for a single element
                The array index
                The path without the array index
        """
        path_bits = path.split("/")
        array_spec = path_bits[-1].find("[")
        
        if array_spec > -1 and path_bits[-1].find("]") == len(path_bits[-1]) - 1:
            array_str = path_bits[-1][array_spec+1:-1]
            
            if array_str.find(":") > -1:
                raise KeyError("Array slices are not supported")
            
            path_bits[-1] = path_bits[-1][:array_spec]
            path_without = ctypes.create_string_buffer(bytes("/".join(path_bits), "utf-8"))
            return (True, int(array_str), path_without)
        
        path_without = ctypes.create_string_buffer(bytes(path, "utf-8"))
        
        return (False, None, path_without)
    
    def _is_list_like(self, value):
        """
        Whether a variable should be treated as a list, in terms of converting
        it to an ODB array.
        
        Args:
            * value (anything)
            
        Returns:
            bool
        """
        allowed_types = [list, tuple, ctypes.Array]

        if have_numpy:
            allowed_types.append(np.ndarray)

        return isinstance(value, tuple(allowed_types))
    
    def _odb_get_content_array_len(self, key_metadata, contents):
        """
        Get the length of ODB array needed to fit the given content. This isn't
        as trivial as just calling len() as we need to consider strings as well.
        
        Args:
            * key_metadata (`midas.structs.Key`)
            * contents (anything)
            
        Returns:
            int
        """
        our_num_values = 1
        
        if self._is_list_like(contents):
            if key_metadata.type == midas.TID_STRING and isinstance(contents, ctypes.Array) and contents._type_ == ctypes.c_char:
                our_num_values = 1
            else:
                our_num_values = len(contents)
        
        return our_num_values
        
    def _odb_resize_array_to_match_content(self, key_metadata, hKey, contents, array_idx=None):
        """
        Resize an array in the ODB to be the correct length for the content 
        we're about to add to it.
        
        Args:
            * key_metadata (`midas.structs.Key`)
            * hKey (int) - ODB key handle
            * contents (anything)
            * array_idx (int) - If setting a single array element, the index
                we're going to set.

        Returns:
            bool - Whether we resized the array
        """
        if array_idx is not None:
            # Don't shrink arrays if setting a single value
            our_num_values = max(key_metadata.num_values, array_idx)
        else:
            our_num_values = self._odb_get_content_array_len(key_metadata, contents)
        
        if our_num_values != key_metadata.num_values:
            self.lib.c_db_set_num_values(self.hDB, hKey, our_num_values)
            return True
        
        return False
    
    def _get_max_str_len(self, contents):
        """
        Get the size of ODB string needed to fit the given content.
        
        Args:
            * contents (string, list of string, etc)
            
        Returns:
            int
        """
        if contents is None:
            our_max_string_size = 32
        elif self._is_list_like(contents):
            if isinstance(contents, ctypes.Array) and contents._type_ == ctypes.c_char:
                # Single c string provided. Add a null byte if needed.
                our_max_string_size = len(contents)
                
                if contents[-1] != b"\x00":
                    our_max_string_size += 1
            else:
                # List of strings provided. Add a null byte.
                our_max_string_size = max(len(x) for x in contents) + 1
        else:
            # Single string provided. Add a null byte.
            our_max_string_size = len(contents) + 1
        
        if our_max_string_size > midas.MAX_STRING_LENGTH:
            our_max_string_size = midas.MAX_STRING_LENGTH
    
        return our_max_string_size
    
    def _odb_lengthen_string_to_fit_content(self, key_metadata, c_path, contents):
        """
        ODB entries for strings have a set capacity. Increase the capacity to 
        fit the content we're about to add to it.
        
        Args:
            * key_metadata (`midas.structs.Key`)
            * c_path (ctypes string buffer) - Path to our ODB entry
            * contents (str, list of str, ctypes string buffer etc)

        Returns:
            bool - Whether we changed the length
        """
        if key_metadata.type != midas.TID_STRING:
            raise TypeError("Dont call _odb_lengthen_string_to_fit_content() on a non-string ODB key")
        
        our_max_string_size = self._get_max_str_len(contents)
        
        if our_max_string_size > key_metadata.item_size:
            self.lib.c_db_resize_string(self.hDB, 0, c_path, key_metadata.num_values, our_max_string_size)
            return True
        
        return False
    
    def _odb_strip_final_slash(self, path):
        """
        Strip the final / from any ODB path, so midas C++ code doesn't get
        confused and start searching for an empty key within the path we're
        searching for.

        E.g. 
        * `/Experiment` => `/Experiment`
        * `/Experiment/` => `/Experiment`
        * `/Experiment/////` => `/Experiment`
        * `/` => `/`

        Args:
            * path (str)

        Returns:
            str
        """
        while path.endswith("/") and path != "/":
            path = path[:-1]
        
        return path

    def _odb_get_type(self, path):
        """
        Get the midas type of the ODB entry at the specified path.
        
        Args:
            * path (str) - The ODB path
            
        Returns:
            int, one of midas.TID_xxx
        """
        return self._odb_get_key(path).type
 
    def _odb_get_hkey(self, path, follow_link=True):
        """
        Get a key handle for the ODB entry at the specified path.
        
        Args:
            * path (str) - The ODB path
            * follow_link (bool)
            
        Returns:
            int
        """
        c_path = ctypes.create_string_buffer(bytes(path, "utf-8"))
        hKey = ctypes.c_int()
        
        if follow_link:
            self.lib.c_db_find_key(self.hDB, 0, c_path, ctypes.byref(hKey))
        else:
            self.lib.c_db_find_link(self.hDB, 0, c_path, ctypes.byref(hKey))
        
        return hKey
        
    def _odb_get_key(self, path, follow_link=True):
        """
        Get metadata about the ODB entry at the specified path.
        
        Args:
            * path (str) - The ODB path
            
        Returns:
            `midas.structs.Key`
        """
        hKey = self._odb_get_hkey(path, follow_link)
        return self._odb_get_key_from_hkey(hKey)
    
    def _odb_get_key_from_hkey(self, hKey):
        """
        Convert a key handle to the actual key metadata.
        
        Args:
            * hKey (int) - from `_odb_get_hkey()`
            
        Returns:
            `midas.structs.Key`            
        """
        key = midas.structs.Key()
        self.lib.c_db_get_key(self.hDB, hKey, ctypes.byref(key))
        return key        
    
    def _odb_get_parent_hkey(self, hKey):
        """
        Convert a key handle to the actual key metadata.
        
        Args:
            * hKey (int) - from `_odb_get_hkey()`
            
        Returns:
            `ctypes.c_int`            
        """
        hParent = ctypes.c_int()
        self.lib.c_db_get_parent(self.hDB, hKey, ctypes.byref(hParent))
        return hParent        
    
    def _odb_enum_key(self, hKey, follow_links=True):
        """
        Get a list of ODB keys beneath this directory.
        
        Args:
            * hKey (`ctypes.c_int`) - From a call to _odb_get_hkey().
            * follow_links (bool) - Whether to follow links or not.
            
        Returns:
            list of 2-tuples of (`ctypes.c_int`, `midas.structs.Key`) for
            hKey, key_metadata.
        """
        subkeys = []
        hSubKey = ctypes.c_int()
        idx = ctypes.c_int()
        idx.value = 0
        
        while True:
            try:
                if follow_links:
                    self.lib.c_db_enum_key(self.hDB, hKey, idx, ctypes.byref(hSubKey))
                else:
                    self.lib.c_db_enum_link(self.hDB, hKey, idx, ctypes.byref(hSubKey))
            except midas.MidasError as e:
                if e.code == midas.status_codes["DB_NO_MORE_SUBKEYS"]:
                    break
                else:
                    raise
            
            if hSubKey == 0:
                break
            
            idx.value += 1
            subkeys.append((hSubKey.value, self._odb_get_key_from_hkey(hSubKey)))
            
        return subkeys

           
    def _odb_get_value(self, path, recurse_dir=False, array_single=False, 
                       include_key_metadata=False, key_metadata=None, hKey=None):
        """
        Get a value from the ODB.
        
        Args:
            * path (str) - The ODB path
            * recurse_dir (bool) - If it's a directory, whether to recurse down and list
                everything.
            * array_single (bool) - If this isn't a directory, and you've already parsed
                the path to see if we expect a single item (user specified /my/path[2] etc).
            * include_key_metadata (bool) - Whether to return just the value, or the metadata 
                and value.
            * key_metadata (`midas.structs.Key`) - If you've already extracted the metadata
                for this path.
            * hKey (`ctypes.c_int`) - If you've already got the hkey for this path.
            
        Returns:
            If include_key_metadata is False, the value.
            If include_key_metadata is True a dict.
        """
        c_path = ctypes.create_string_buffer(bytes(path, "utf-8"))
        
        if hKey is None:
            hKey = ctypes.c_int()
            self.lib.c_db_find_key(self.hDB, 0, c_path, ctypes.byref(hKey))

        if key_metadata is None:
            key_metadata = self._odb_get_key_from_hkey(hKey)
        
        array_len = key_metadata.num_values
        
        if array_single:
            array_len = 1
        
        if key_metadata.type == midas.TID_STRING:
            str_len = key_metadata.item_size
        else:
            str_len = None
                
        if key_metadata.type == midas.TID_KEY:
            # A directory
            child_keys = self._odb_enum_key(hKey, False)
            retval = collections.OrderedDict()
            
            for (child_hkey, child_key) in child_keys:
                child_name_data = "%s" % child_key.name.decode("utf-8")
                child_name_meta = "%s/key" % child_name_data
                
                if recurse_dir or child_key.type != midas.TID_KEY:
                    path_slash = "%s/" % path if not path.endswith("/") else path
                    child_path = "%s%s" % (path_slash, child_key.name.decode("utf-8"))

                    if child_key.type == midas.TID_LINK:
                        child_path = self.odb_get_link_destination(child_path)
                        child_data = self._odb_get_value(child_path, 
                                                recurse_dir=recurse_dir, 
                                                include_key_metadata=include_key_metadata)
                    else:
                        child_data = self._odb_get_value(child_path,                                                     
                                                recurse_dir=recurse_dir, 
                                                include_key_metadata=include_key_metadata, 
                                                key_metadata=child_key, 
                                                hKey=child_hkey)

                    if include_key_metadata:
                        retval[child_name_data] = child_data[child_name_data]
                    else:
                        retval[child_name_data] = child_data
                else:
                    # No recursion - just show an empty dict
                    retval[child_name_data] = collections.OrderedDict()
                    
                if include_key_metadata:
                    retval[child_name_meta] = self._odb_key_metadata_to_dict(child_key)
        else:
            # Not a directory.
            # Get a ctypes object for midas to put the data into.
            c_rdb = self._midas_type_to_ctype(key_metadata.type, array_len, str_len=str_len, odb_path=path)
            c_size = ctypes.c_int(ctypes.sizeof(c_rdb))
            
            self.lib.c_db_get_value(self.hDB, 0, c_path, ctypes.byref(c_rdb), ctypes.byref(c_size), key_metadata.type, 0)

            if key_metadata.type == midas.TID_STRING:
                if array_len > 1:
                    # Arrays of string are the nastiest bit, as we only given one
                    # "mega-string" containing all the entries which we now need to
                    # split up.
                    # This line splits the string into sections of correct length; 
                    # then for each line it removes everything after the first null 
                    # byte, then decode into a python string. We need to do it this 
                    # way (rather than simply splitting c_rdb.value) as otherwise 
                    # we'd just get null for everything after the end of the first 
                    # entry in the array.
                    retval = [c_rdb[i * str_len:(i+1) * str_len].split(b'\0',1)[0].decode("utf-8") for i in range(array_len)]
                else:
                    retval = c_rdb.value.decode("utf-8")
            else:
                if array_len > 1:
                    # Convert to a regular list.
                    retval = [x for x in c_rdb]
                else:
                    retval = c_rdb.value
                    
        if include_key_metadata:
            # Normally we just return a single value, but if the user
            # wants the metadata as well, we need to mockup a dict
            # containing the relevant information.
            name_data = "%s" % key_metadata.name.decode("utf-8")
            name_meta = "%s/key" % name_data
            retval = {name_data: retval}
            retval[name_meta] = self._odb_key_metadata_to_dict(key_metadata)
                
        return retval

    def _odb_key_metadata_to_dict(self, key_metadata):
        """
        Convert ODB key metadata to a dict like used by db_json_ls().
        
        Args:
            * key_metadata (`midas.structs.Key`)
            
        Returns:
            dict
        """
        retval = {"type": key_metadata.type,
                             "access_mode": key_metadata.access_mode,
                             "last_written": key_metadata.last_written}
        
        if key_metadata.type == midas.TID_STRING:
            retval["item_size"] = key_metadata.item_size
        if key_metadata.num_values > 1:
            retval["num_values"] = key_metadata.num_values
            
        return retval
                    
    def _odb_fix_directory_order(self, path, hKey, contents):
        """
        ODB directories have a well-defined order. If the user gave an
        ordered dictionary, this function ensures that the ODB order matches
        the order the user specified.
        
        Args:
            * path (str) - The ODB path
            * hKey (int) - from `_odb_get_hkey()`
            * contents (`collections.OrderedDict`)
        """
        current_order = self.odb_get(path, just_key_list=True)
        target_order = [k for k in contents.keys()]
        
        if len(current_order) != len(target_order):
            # User only updated some of the values, and didn't specify others.
            # By definition we don't have enough information to determine the
            # correct order.
            return
        
        if not path.endswith("/"):
            path += "/"
        
        # We do this rather simplistically and just find the first 
        # ODB entry that's in the wrong place. Then set the index
        # for all the entries after that one.
        force_fix = False
        
        for i in range(len(target_order)):
            if force_fix or current_order[i] != target_order[i]:
                force_fix = True
                subkey = self._odb_get_hkey(path + target_order[i])
                self.lib.c_db_reorder_key(self.hDB, subkey, i)
        
    def _type_to_str(self, objtype):
        return str(objtype).replace("<class ", "").replace(">", "")

    def _midas_type_to_ctype(self, midas_type, array_len=None, str_len=None, initial_value=None, odb_path=None):
        """
        Get the appropriate ctype object for the given midas type, optionally
        setting the value based on a given python value.
        
        Args:
            * midas_type (int) - One of midas.TID_xxx
            * array_len (int) - Explicit array length. If None, but initial_value
                is not None, it's taken from the size needed for initial_value.
                If array_len and initial_value are None, it's 1.
            * str_len (int) - Explicit string length if creating a ctypes string 
                buffer for a midas.TID_STRING. If None, but initial_value
                is not None, it's taken from the size needed for initial_value.
                If str_len and initial_value are None, it's 32.
            * initial_value (anything)
            * odb_path (string) - This ODB path will be prefixed to any exceptions 
                we raise to provide more readable error messages.
            
        Returns:
            ctypes.c_int8, ctypes.c_uint32 etc....
        """
        err_prefix = "" if odb_path is None else "%s: " % odb_path

        if array_len is None:
            if self._is_list_like(initial_value): 
                array_len = len(initial_value)
                
                if array_len == 1:
                    # Convert arrays of length 1 to single values for easier handling
                    # later.
                    initial_value = initial_value[0]
            else:
                array_len = 1

        if midas_type == midas.TID_STRING:
            if str_len is None:
                str_len = self._get_max_str_len(initial_value)
                
            total_str_len = str_len * array_len
            total_initial_value = None

            if initial_value is None: 
                total_initial_value = b""
            else:
                if isinstance(initial_value, ctypes.Array):
                    total_initial_value = initial_value.value
                    # We previously assumed this was a list, but it's really
                    # just one string that was passed in as an array fo chars.
                    # Update the total string length to reflect that.
                    total_str_len = str_len
                elif self._is_list_like(initial_value):
                    total_initial_value = b""
                    
                    for this_val in initial_value:
                        add_val = this_val[:str_len]
                        if isinstance(add_val, bytes):
                            total_initial_value += add_val
                        else:
                            total_initial_value += bytes(add_val, "utf-8")
                        total_initial_value += bytes(str_len - len(add_val)) # Pad with null bytes
                elif isinstance(initial_value, bytes):
                    total_initial_value = initial_value[:str_len]
                else:
                    total_initial_value = bytes(initial_value[:str_len], "utf-8")
            
            if total_initial_value is not None:
                retval = ctypes.create_string_buffer(total_initial_value, total_str_len)
                
            return retval
        
        
        retval = None
        rettype = None
        casttype = None

        if midas_type == midas.TID_BYTE or midas_type == midas.TID_CHAR:
            rettype = ctypes.c_ubyte
            casttype = int
        elif midas_type == midas.TID_SBYTE:
            rettype = ctypes.c_byte
            casttype = int
        elif midas_type == midas.TID_WORD:
            rettype = ctypes.c_ushort
            casttype = int
        elif midas_type == midas.TID_SHORT:
            rettype = ctypes.c_short
            casttype = int
        elif midas_type == midas.TID_DWORD or midas_type == midas.TID_BITFIELD:
            rettype = ctypes.c_uint
            casttype = int
        elif midas_type == midas.TID_INT:
            rettype = ctypes.c_int
            casttype = int
        elif midas_type == midas.TID_QWORD:
            rettype = ctypes.c_uint64
            casttype = int
        elif midas_type == midas.TID_INT64:
            rettype = ctypes.c_int64
            casttype = int
        elif midas_type == midas.TID_BOOL:
            rettype = ctypes.c_uint
            casttype = int
        elif midas_type == midas.TID_FLOAT:
            rettype = ctypes.c_float
            casttype = float
        elif midas_type == midas.TID_DOUBLE:
            rettype = ctypes.c_double
            casttype = float
        else:
            raise NotImplementedError("%sRequested midas type (%s) not handled yet" % (err_prefix, midas_type))
                
        if array_len is None or array_len == 1:
            if isinstance(initial_value, ctypes._SimpleCData):
                if isinstance(initial_value, rettype):
                    retval = initial_value
                else:
                    if midas_type == midas.TID_BOOL and isinstance(initial_value, ctypes.c_bool):
                        # Bools are special - users can provide a c_bool but
                        # we actually need to give a c_uint32 to midas...
                        retval = rettype(int(initial_value.value))
                    else:
                        raise TypeError("%sExpected a %s but you provided a %s" % (err_prefix, self._type_to_str(rettype), self._type_to_str(type(initial_value))))
            else:
                retval = rettype()
                
                if initial_value is not None:
                    if isinstance(initial_value, str) and initial_value.startswith("0x"):
                        retval.value = casttype(initial_value, 16)
                    else:
                        retval.value = casttype(initial_value)
        else:
            rettype_arr = rettype * array_len
            
            if initial_value is None:
                retval = rettype_arr()
            elif isinstance(initial_value, ctypes.Array):
                if isinstance(initial_value, rettype_arr):
                    # Already provided a c_int * 4, for example.
                    retval = initial_value
                else:
                    if midas_type == midas.TID_BOOL and isinstance(initial_value[0], ctypes.c_bool) or initial_value[0] is True or initial_value[0] is False:
                        pyint = [int(x) for x in initial_value]
                        retval = rettype_arr(*pyint)
                    else:
                        raise TypeError("%sExpected a %s but you provided a %s" % (err_prefix, self._type_to_str(rettype_arr), self._type_to_str(type(initial_value))))
            elif isinstance(initial_value[0], ctypes._SimpleCData):
                for i, v in enumerate(initial_value):
                    if not isinstance(v, rettype):
                        if midas_type == midas.TID_BOOL and isinstance(v, ctypes.c_bool):
                            initial_value[i] = True if v else False
                        else:
                            raise TypeError("%sExpected a %s but you provided a %s" % (err_prefix, self._type_to_str(rettype), self._type_to_str(type(v))))
                retval = rettype_arr(*initial_value)
            elif have_numpy and isinstance(initial_value, np.ndarray) and isinstance(initial_value[0], midas.tid_np_formats[midas_type]):
                retval = initial_value.ctypes.data_as(ctypes.POINTER(rettype_arr)).contents
            else:
                if len(initial_value) and isinstance(initial_value[0], str) and initial_value[0].startswith("0x"):
                    castlist = [casttype(x, 16) for x in initial_value]
                else:
                    castlist = [casttype(x) for x in initial_value]
                retval = rettype_arr(*castlist)
            
        return retval
    
    def _ctype_to_midas_type(self, value):
        """
        Convert from a ctypes object to the appropriate midas.TID_xxx type.
        
        Args:
            * value (anything)
            
        Returns:
            int, one of midas.TID_xxx
        """
        if isinstance(value, ctypes.Array):
            value = value._type_()
            
            if isinstance(value, ctypes.c_char):
                return midas.TID_STRING
        elif isinstance(value, list) or isinstance(value, tuple):
            if len(value) == 0:
                raise ValueError("Can't determine type of 0-length array")
            value = value[0]
        elif have_numpy and isinstance(value, np.ndarray):
            if len(value) == 0:
                raise ValueError("Can't determine type of 0-length array")
            value = value[0]

            for midas_tid, np_dtype in midas.tid_np_formats.items():
                if np_dtype is not None and isinstance(value, np_dtype):
                    return midas_tid
                        
        if value is True or value is False or isinstance(value, ctypes.c_bool):
            # Note the order here is important isinstance(True, int) also returns True!
            return midas.TID_BOOL
        if isinstance(value, int) or isinstance(value, ctypes.c_int):
            return midas.TID_INT
        if isinstance(value, float) or isinstance(value, ctypes.c_double):
            return midas.TID_DOUBLE
        if isinstance(value, ctypes.c_float):
            return midas.TID_FLOAT
        if isinstance(value, str) or (isinstance(value, ctypes.Array) and isinstance(value._type_(), ctypes.c_char)):
            return midas.TID_STRING
        if isinstance(value, dict):
            return midas.TID_KEY
        if isinstance(value, ctypes.c_ubyte):
            return midas.TID_BYTE
        if isinstance(value, ctypes.c_byte):
            return midas.TID_SBYTE
        if isinstance(value, ctypes.c_ushort):
            return midas.TID_WORD
        if isinstance(value, ctypes.c_short):
            return midas.TID_SHORT
        if isinstance(value, ctypes.c_uint):
            return midas.TID_DWORD
        if isinstance(value, ctypes.c_int64):
            return midas.TID_INT64
        if isinstance(value, ctypes.c_uint64):
            return midas.TID_QWORD
        if isinstance(value, dict):
            return midas.TID_KEY
    
        raise TypeError("Couldn't find an appropriate midas type for value of type %s" % type(value))
    
    def _run_transition(self, trans, run_num, async_flag):
        c_trans = ctypes.c_int(trans)
        c_run_num = ctypes.c_int(run_num)
        c_str = ctypes.create_string_buffer(256)
        c_str_len = ctypes.c_int(ctypes.sizeof(c_str))
        c_async = ctypes.c_int(midas.TR_ASYNC if async_flag else midas.TR_SYNC)
        c_debug = ctypes.c_int(0)
        retval = self.lib.c_cm_transition(c_trans, c_run_num, c_str, c_str_len, c_async, c_debug)
        
        if retval != midas.status_codes["SUCCESS"]:
            py_str = c_str.value.decode("utf-8")
            
            if retval == 110:
                raise midas.TransitionDeferredError(retval, py_str)
            else:
                raise midas.TransitionFailedError(retval, py_str)
        
        return retval

    def _hist_validate_event_name(self, event_name):
        events = self.hist_get_events()

        if event_name not in events:
            midas.status_code_to_exception(midas.status_codes["HS_UNDEFINED_EVENT"], [])

    def _hist_validate_tag_name(self, tag_name, tags):
        valid_tag = False

        for tag in tags:
            if tag["name"] == tag_name:
                valid_tag = True
                break

        if not valid_tag:
            midas.status_code_to_exception(midas.status_codes["HS_UNDEFINED_VAR"], [])
