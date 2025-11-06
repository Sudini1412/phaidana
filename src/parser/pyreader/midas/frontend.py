"""
This module provides a way to write midas frontends in python.

It supports many of the features of the C frontend (including the concepts
of "Equipment"), with a few limitations.

There are 2 main classes you need to create derived versions of:
    * EquipmentBase
    * FrontendBase

See full examples in `examples/basic_frontend.py` and `examples/multi_frontend.py`.

Limitations:
    * Only supports periodic and polled equipment (no user-triggered etc).
    * Doesn't give an option for automatic daemonization (no -D flag).
    * Only supports writing events in MIDAS format, not FIXED format.
    * Multi-threaded support is untested and not guaranteed.
"""

import midas
import midas.client
import midas.event
import socket
import os.path
import datetime
import time
import ctypes
import argparse
import collections
import logging
import sys

frontend_index = None    # If run with the -i flag on the command-line. Populated by parse_args().
cmd_line_hostname = None # If run with the -h flag on the command-line. Populated by parse_args().
cmd_line_exptname = None # If run with the -e flag on the command-line. Populated by parse_args().
daemon_flag = None       # None/0/1 for not daemon/-D/-O.
args_parsed = False      # Whether parse_args() has been called or not.

logger = logging.getLogger('midas')

# mfe uses -h for host rather than help
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-i", type=int, metavar="frontend_index")
parser.add_argument("-h", type=str, metavar="host_name")
parser.add_argument("-e", type=str, metavar="expt_name")
parser.add_argument("-d", action="store_true", help="Debug")
parser.add_argument("-D", action="store_true", help="Become a daemon")
parser.add_argument("-O", action="store_true", help="Become a daemon but retain stdout")
parser.add_argument('--help', action='help', help='Show this help message and exit')

def parse_args():
    """
    Parse command-line arguments. Specifically we support:
        * -i for the frontend index
        * -h for the midas hostname (defaults to what's set by env vars)
        * -e for the midas expt name (defaults to what's set by env vars)
        * -d to enable debug logging
        * -D to become a daemon
        * -O to become a daemon but retain stdout
        
    If you want to support more arguments for your frontend, you can work on
    the global `parser` object and call add_argument() etc. For example:
    
    ```
    parser = midas.frontend.parser
    parser.add_argument("-x", type=int, help="Some custom argument")
    
    args = midas.frontend.parse_args()
    
    print("X argument was %s" % args.x)
    
    # Presumably you will use the argument for something in your frontend class.
    my_fe = SomeFrontendClass(args.x)
    my_fe.run()
    ```
    
    Returns:
        The result of calling parser.parse_args()
    """
    global frontend_index, cmd_line_hostname, cmd_line_exptname, args_parsed, parser, daemon_flag
    
    args = parser.parse_args()
    
    if args.d:
        formatter = logging.Formatter("%(asctime)s (%(filename)s:%(lineno)d:%(funcName)s) - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    frontend_index = args.i if args.i is not None else -1
    cmd_line_hostname = args.h
    cmd_line_exptname = args.e
    daemon_flag = None
    
    if args.D and args.O:
        raise ValueError("Only -D or -O may be specified, not both")
    elif args.D:
        daemon_flag = 0
    elif args.O:
        daemon_flag = 1
    
    logger.info("cmd_line_hostname: %s" % cmd_line_hostname)
    logger.info("cmd_line_exptname: %s" % cmd_line_exptname)
    logger.info("frontend_index:    %s" % frontend_index)
    
    args_parsed = True
    return args

class InitialEquipmentCommon:
    """
    Equipment settings are stored in the ODB at /Equipment/<xxx>/Common.
    But the first time we run a frontend/equipment we need to populate the ODB.
    This class/object defines the parameters the develop should set.
    
    At a minimum, the developer must update `equip_type` and `event_id`.
    
    Members:
        * equip_type (int) - Whether this is a periodic or polled equipment.
            One of midas.EQ_xxx (or a bitmask from multiple such flags).
        * event_id (int) - The "Event ID" events from this equipment will be
            flagged with by default. Each equipment in your experiment should
            have a different event ID.
        * buffer_name (str) - Name of the buffer events from this equipment
            will be sent to. The framework will create the buffer for you if
            needed.
        * trigger_mask (int) - The "Trigger Mask" events from this equipment will
            be flagged with by default.
        * period_ms (float) - How often the readout function will be called for
            periodic equipment.
        * read_when (int) - When to call the readout function of this equipment.
            One of midas.RO_xx (or a bitmask from multiple such flags).
        * log_history (bool) - Whether to periodically copy events produced by
            this equipment into the ODB (in /Equipment/<xxx>/Variables) so they
            get logged in the history database.
    """
    def __init__(self):
        self.equip_type = None
        self.event_id = None
        self.buffer_name = ""
        self.trigger_mask = 0
        self.period_ms = 1000
        self.read_when = midas.RO_RUNNING
        self.log_history = False

class EquipmentBase:
    """
    You should define classes that inherit from this base class.
    
    There are 2 main types of equipment:
        * Periodic equipment (midas.EQ_PERIODIC) - The frontend will call
            readout_func() periodically. readout_func() should return an
            event, or None if no event should be sent.
        * Polled equipment (midas.EQ_POLLED) - The frontend will call
            poll_func() often, which says whether an event is ready or not;
            if an event is ready, the frontend will then call readout_func().
    
    In your derived class you should implement the following functions:
        * __init__(), which calls this base class' __init__()
        * readout_func() that creates and returns a `midas.event.Event`
        * poll_func() if you are creating a polled equipment
        
    See the documentation of those functions for more details, and the
    examples in the examples/ directory.
    
    Members:
        * client (`midas.client.MidasClient`) - Client that you can use from
            within your equipment's functions to access the ODB etc.
        * name (str) - Name of this equipment.
        * frontend_name (str) - Name of the frontend/client that created this
            equipment.
        * settings (dict) - The content of /Equipment/<xxx>/Settings (or None
            if not settings exist). Updated automatically when there are 
            changes in the ODB.
        * common (dict) - The content of /Equipment/<xxx>/Common. Updated
            automatically when there are changes in the ODB.
        * stats (dict) - Number of events this equipment has sent, etc.
        * last_get_event_time (datetime.datetime) - When we last tried to get
            an event from this equipment.
        * odb_var_write_period (datetime.timedelta) - How often we should copy
            an event into the ODB for this equipment.
        * last_odb_var_write_time (datetime.datetime) - When we last copied an
            event into the ODB from this equipment.
        * odb_common_dir (str) - /Equipment/<xxx>/Common
        * odb_stats_dir (str) - /Equipment/<xxx>/Statistics
        * odb_variables_dir (str) - /Equipment/<xxx>/Variable
        * odb_settings_dir (str) - /Equipment/<xxx>/Settings
        * _inited (bool) - Whether __init__() has been called
    """
    def __init__(self, midas_client, equip_name, default_common, default_settings = None, overwrite_common = False):
        """
        You MUST call this function from your derived class' constructor.
        
        It sets up the relevant entries for this equipment in the ODB, sets up
        hotlinks so we know when /Equipment/<xxx>/Common or 
        /Equipment/<xxx>/Settings change, and sets the initial "status" of this
        equipment.
        
        Args:
            * midas_client (`midas.client.MidasClient`) - The object we can use
                for ODB access etc.
            * equip_name (str) - The name of this equipment. It should be
                unique. We do NOT do any automatic substitution of the frontend 
                index etc.
            * default_common (`InitialEquipmentCommon`) - The initial settings
                we should use to populate /Equipment/<xxx>/Common the first 
                time this equipment is run. After that, the ODB settings take
                precendence unless you also pass overwrite_common=True.
            * default_settings (dict, `collection.OrderedDict`, or None) - Any 
                ODB parameters we should create in /Equipment/<xxx>/Settings
                if that directory does not already exist. We do NOT update the
                values of any ODB keys that already exist. However, we will 
                add/remove ODB keys if the structure you pass in does not match
                the existing ODB structure. This allows you to easily add/remove
                settings as your frontend development progresses.
            * overwrite_common (bool) - If True, we'll overwrite 
                /Equipment/<xxx>/Common with the values in the code, overwriting
                any changes that the user has made in the ODB since the frontend was
                last started. This is equivalent to setting 
                `BOOL equipment_common_overwrite = TRUE;` in mfe C++ frontends.
        """
        if self._is_initialized():
            logger.debug("Equipment %s is already initialized" % equip_name)
            return
        
        if not isinstance(default_common, InitialEquipmentCommon):
            raise TypeError("default_settings should be a InitialEquipmentCommon object")
        
        if default_common.equip_type is None or default_common.event_id is None:
            raise ValueError("default_settings must have equip_type and event_id set")
        
        logger.info("Initializing equipment %s" % equip_name)
        self.name = equip_name
        self.common = None
        self.stats = {"events": 0,
                      "kbytes": 0,
                      "last_events": 0,
                      "last_time": None}
        
        self.last_get_event_time = None
        
        self.odb_var_write_period = datetime.timedelta(milliseconds=1000)
        self.last_odb_var_write_time = None
        
        self.odb_common_dir = "/Equipment/%s/Common" % self.name
        self.odb_stats_dir = "/Equipment/%s/Statistics" % self.name
        self.odb_variables_dir = "/Equipment/%s/Variables" % self.name
        self.odb_settings_dir = "/Equipment/%s/Settings" % self.name
        
        self.client = midas_client
        self.frontend_name = self.client.name

        if overwrite_common or not self.client.odb_exists(self.odb_common_dir):
            common = collections.OrderedDict([
                      ("Event ID", ctypes.c_ushort(default_common.event_id)),
                      ("Trigger mask", ctypes.c_ushort(default_common.trigger_mask)),
                      ("Buffer", ctypes.create_string_buffer(bytes(default_common.buffer_name, "utf-8"), 32)),
                      ("Type", ctypes.c_int32(default_common.equip_type)),
                      ("Source", ctypes.c_int32(0)),
                      ("Format", ctypes.create_string_buffer(b"MIDAS", 8)),
                      ("Enabled", True),
                      ("Read on", ctypes.c_int32(default_common.read_when)),
                      ("Period", ctypes.c_int32(default_common.period_ms)),
                      ("Event limit", ctypes.c_double(0)),
                      ("Num subevents", ctypes.c_uint32(0)),
                      ("Log history", ctypes.c_int32(default_common.log_history)),
                      ("Frontend host", ctypes.create_string_buffer(bytes(socket.gethostname(), "utf-8")[:31], 32)),
                      ("Frontend name", ctypes.create_string_buffer(bytes(self.frontend_name, "utf-8")[:31], 32)),
                      ("Frontend file name", ctypes.create_string_buffer(bytes(os.path.basename(sys.argv[0]), "utf-8")[:255], 256)),
                      ("Status", ctypes.create_string_buffer(256)),
                      ("Status color", ctypes.create_string_buffer(32)),
                      ("Hidden", False),
                      ("Write cache size", ctypes.c_int32(100000))
                ])
            
            logger.info("Setting up %s" % self.odb_common_dir)
            self.client.odb_set(self.odb_common_dir, common)
            
        self.set_status("%s@%s" % (self.frontend_name, socket.gethostname()), "greenLight")
        
        self.common = self.client.odb_get(self.odb_common_dir)
        self.client.odb_watch(self.odb_common_dir, self._odb_common_callback)
        
        if default_settings is not None:
            logger.info("Setting up %s" % self.odb_settings_dir)
            self.client.odb_set(self.odb_settings_dir, default_settings, update_structure_only=True)
                
        if self.client.odb_exists(self.odb_settings_dir):
            self.settings = self.client.odb_get(self.odb_settings_dir)
            self.client.odb_watch(self.odb_settings_dir, self._odb_settings_callback, True)
        else:
            self.settings = None
        
        self._update_statistics_in_odb() # Reset to 0 (as self.stats was zeroed above)
        
        if not self.client.odb_exists(self.odb_variables_dir):
            logger.info("Setting up %s for the first time" % self.odb_variables_dir)
            self.client.odb_set(self.odb_variables_dir, {})

        self._inited = True

    def poll_func(self):
        """
        If you are creating a polled equipment (midas.EQ_POLLED), you MUST 
        implement this function in your derived class.
        
        It should return True if you want readout_func() to be called (which
            will then construct and return the actual event).
        It should return False if no event should be read out.
        """
        raise NotImplementedError("You should implement poll_func in your polled equipment class!")
    
    def readout_func(self):
        """
        You MUST implement this function in your derived class if you ever want
        to send events.
        
        It should return None if you don't want to send an event at this time.
        It should return a `midas.event.Event` if you do want to send an event;
            the frontend will convert it to the binary format midas expects
            automatically. 

        If you're pushing the midas python code to its limits, you may return
        a list of `midas.event.Event`. For example, if you're using a polled
        frontend and have a backlog of 5 events to send, it will be more
        efficient to send all 5 in one call to `readout_func()` rather than
        waiting for `poll_func()/readout_func()` to be called 5 times each.
        """
        raise NotImplementedError("You should implement readout_func in your equipment class!")
    
    def settings_changed_func(self):
        """
        You MAY implement this function in your derived class if you want to be
        told when a variable in /Equipment/<xxx>/Settings has changed. 
        
        If you want more fine-grained information about WHAT changed (not just
        that some setting changed), implement `detailed_settings_changed_func()`
        instead.
        
        * `settings_changed_func()` => High-level "something changed"
        * `detailed_settings_changed_func()` => Low-level "this specific thing changed"
        
        We will automatically update self.settings before calling this function,
        but you may want to implement this function to validate any settings
        the user has set, for example.
        
        It may return anything (we don't check it).
        """
        pass
    
    def detailed_settings_changed_func(self, path, idx, new_value):
        """
        You MAY implement this function in your derived class if you want to be
        told when a variable in /Equipment/<xxx>/Settings has changed.
        
        If you don't care about what changed (just that any setting has changed),
        implement `settings_changed_func()` instead,
        
        We will automatically update self.settings before calling this function,
        but you may want to implement this function to validate any settings
        the user has set, for example.
        
        It may return anything (we don't check it).
        
        Args:
            * path (str) - The ODB path that changed
            * idx (int) - If the entry is an array, the array element that changed
            * new_value (int/float/str etc) - The new ODB value (the single array
                element if it's an array that changed)
        """
        pass
    
    def set_status(self, status_text, status_color="greenLight"):
        """
        You can call this function to set the "status" of this equipment.
        The status is shown on the main midas status page.
        
        Args:
            * status_text (str) - This should be a short string, or else the
                formatting of the main status page may get a little wacky.
            * status_color (str) - There are 4 pre-defined colors (greenLight, 
                yellowLight, yellowGreenLight, redLight) or you can set any
                CSS background color name / hex value (max length 32 chars).
        """
        logger.debug("Setting status of equipment %s to `%s` (color `%s`)", self.name, status_text, status_color)
        self.client.odb_set("%s/%s" % (self.odb_common_dir, "Status"), ctypes.create_string_buffer(bytes(status_text, "utf-8"), 256))
        self.client.odb_set("%s/%s" % (self.odb_common_dir, "Status color"), ctypes.create_string_buffer(bytes(status_color, "utf-8"), 32))

    def _is_initialized(self):
        """
        Whether the base class __init__() has been called.
        
        Returns:
            bool
        """
        return hasattr(self, "_inited")

    def _is_history_logging_enabled(self):
        """
        Whether we should be copying events from this equipment into the ODB.
        
        Returns:
            bool
        """
        return (self.common["Read on"] & midas.RO_ODB) or self.common["Log history"]
            
    def _is_active_for_state(self, run_state):
        """
        Whether we should call readout_func() for this equipment in the given
        run state.
        
        Args:
            * run_state (int) - One of midas.STATE_xxx
            
        Returns:
            bool
        """
        if run_state == midas.STATE_RUNNING and self.common["Read on"] & midas.RO_RUNNING:
            return True
        if run_state == midas.STATE_STOPPED and self.common["Read on"] & midas.RO_STOPPED:
            return True
        if run_state == midas.STATE_PAUSED and self.common["Read on"] & midas.RO_PAUSED:
            return True
        
        return False
            
    def _is_active_for_transition(self, transition):
        """
        Whether we should call readout_func() for this equipment at the given 
        run transition.
        
        Args:
            * transition (int) - One of midas.TR_xxx
            
        Returns:
            bool
        """
        if transition == midas.TR_START and self.common["Read on"] & midas.RO_BOR:
            return True
        if transition == midas.TR_STOP and self.common["Read on"] & midas.RO_EOR:
            return True
        if transition == midas.TR_PAUSE and self.common["Read on"] & midas.RO_PAUSE:
            return True
        if transition == midas.TR_RESUME and self.common["Read on"] & midas.RO_RESUME:
            return True
        
        return False
            
    def _stats_add_event_of_size(self, size_bytes):
        """
        Record that we've sent an event of the given size. This only updates
        our local record, not the ODB version (call _update_statistics_in_odb())
        to do that.
        
        Args:
            * size_bytes (int)
        """
        self.stats["events"] += 1
        self.stats["kbytes"] += size_bytes / 1024
    
    def _reset_stats(self):
        """
        Reset the number/size of events this equipment has sent to 0.
        """
        self.stats["last_time"] = None
        self.stats["kbytes"] = 0
        self.stats["events"] = 0
        self.stats["last_events"] = 0
        self._update_statistics_in_odb()
            
    def _update_statistics_in_odb(self):
        """
        Update /Equipment/<xxx>/Statistics in the ODB with the total number
        of events sent, and recent event/data rates.
        """
        now = datetime.datetime.now()
        
        if self.stats["last_time"] is None:
            secs = 1
        else:
            secs = (now - self.stats["last_time"]).total_seconds()
            
        ev_per_sec = (self.stats["events"] - self.stats["last_events"]) / secs
        kB_per_sec = self.stats["kbytes"] / secs
        
        c_stats = collections.OrderedDict([("Events sent", ctypes.c_double(self.stats["events"])),
                                           ("Events per sec.", ctypes.c_double(ev_per_sec)),
                                           ("kBytes per sec.", ctypes.c_double(kB_per_sec))])
        
        self.client.odb_set(self.odb_stats_dir, c_stats)
        self.stats["last_events"] = self.stats["events"]
        self.stats["last_time"] = now
        self.stats["kbytes"] = 0
            
    def _odb_common_callback(self, client, path, new_value):
        """
        Callback function we register to be told when /Equipment/<xxx>/Common
        changes. We do some sanity-checking and warn the user if there are
        problems.
        
        Args:
            * client - Needed to match the expected interface
            * path - Needed to match the expected interface
            * new_value (dict) - The new /Common values
        """
        logger.debug("Common settings have changed")
        
        if new_value["Buffer"] != self.common["Buffer"]:
            self.client.msg("You must restart the frontend to update which buffer is written to")
            
        if new_value["Num subevents"] != 0:
            self.client.msg("Python frontends don't support subevents", True)
        
        if new_value["Format"] != "MIDAS":
            self.client.msg("Python frontends only support 'MIDAS' format events", True)
            self.client.odb_set(self.odb_common_dir + "/Format", "MIDAS")
            
        if new_value["Type"] & midas.EQ_MANUAL_TRIG:
            self.client.msg("Python frontends don't support manually-triggered events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_MANUAL_TRIG)
            
        if new_value["Type"] & midas.EQ_FRAGMENTED:
            self.client.msg("Python frontends don't support fragmented events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_FRAGMENTED)
            
        if new_value["Type"] & midas.EQ_INTERRUPT:
            self.client.msg("Python frontends don't support interrupt events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_INTERRUPT)
            
        if new_value["Type"] & midas.EQ_MULTITHREAD:
            self.client.msg("Python frontends don't support multithreaded events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_MULTITHREAD)
            
        if new_value["Type"] & midas.EQ_SLOW:
            self.client.msg("Python frontends don't support slow control events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_SLOW)
            
        if new_value["Type"] & midas.EQ_USER:
            self.client.msg("Python frontends don't support user events", True)
            self.client.odb_set(self.odb_common_dir + "/Type", new_value["Type"] & ~midas.EQ_USER)
            
        self.common = new_value
            
    def _odb_settings_callback(self, client, path, idx, new_value):
        """
        Callback function we register to be told when /Equipment/<xxx>/Settings
        changes. The user should not override this function, but can override
        the `settings_changed_func()` function we call from within it.
        
        Args:
            * client - Needed to match the expected interface
            * path - Needed to match the expected interface
            * new_value (dict) - The new /Settings values
        """
        logger.debug("User's settings have changed")
        self.settings = self.client.odb_get(self.odb_settings_dir)
        
        # High-level "something changed" function
        self.settings_changed_func()
        
        # Low-level "this thing changed" function
        self.detailed_settings_changed_func(path, idx, new_value)
        
    def _is_event_limit_reached(self):
        """
        If the user has set a limit of the number of events in 
        /Equipment/<xxx>/Common/Event limit, whether we've reached that limit.
        
        Returns:
            bool
        """
        if self.common["Event limit"] <= 0 or self.stats is None:
            return False
        
        return self.stats["events"].value >= self.common["Event limit"]
    
    def _is_time_to_write_vars_to_odb(self):
        """
        Whether we should copy the next event to /Equipment/<xxx>/Variables.
        
        Returns:
            bool
        """
        if not self._is_history_logging_enabled():
            return False
        
        if self.last_odb_var_write_time is None:
            return True
        
        return datetime.datetime.now() - self.last_odb_var_write_time > self.odb_var_write_period
    
    def _is_time_to_work(self):
        """
        For a periodic frontend, whether it's time to call readout_func().
        
        Returns:
            bool
        """
        if self.last_get_event_time is None:
            return True
        
        threshold = datetime.timedelta(milliseconds=self.common["Period"])
        
        return datetime.datetime.now() - self.last_get_event_time > threshold
    
    def _is_periodic(self):
        """
        Whether this is a periodic frontend.
        
        Returns:
            bool
        """
        return self.common["Type"] & midas.EQ_PERIODIC
    
    def _is_polled(self):
        """
        Whether this is a polled frontend.
        
        Returns:
            bool
        """
        return self.common["Type"] & midas.EQ_POLLED
    
    def _get_events(self):
        """
        Call the user's readout_func(), set any header elements they haven't
        set, and return the event.
        
        Returns:
            list of `midas.event.Event`
        """
        self.last_get_event_time = datetime.datetime.now()
        events = self.readout_func()
        
        if not isinstance(events, list):
            events = [events]

        non_empty_events = [e for e in events if e is not None]

        if len(non_empty_events) == 0:
            return []
        
        for i, event in enumerate(non_empty_events):
            if not isinstance(event, midas.event.Event):
                raise TypeError("Expected a midas.event.Event")
                    
            if event.header.event_id is None:
                event.header.event_id = self.common["Event ID"]
                
            if event.header.trigger_mask is None:
                event.header.trigger_mask = self.common["Trigger mask"]
                
            if event.header.serial_number is None:
                event.header.serial_number = self.stats["events"] + i
                
            if event.header.timestamp is None:
                event.header.timestamp = int(time.time())
                
        return non_empty_events

class FrontendBase:
    """
    A frontend can control multiple equipment. You should define a class that 
    inherits from this base class.
    
    Configuration of the frontend/equipment should be done in your class'
    __init__(). It should:
        * Call this base class' __init__
        * Create and add any equipment.
        
    You may implement the begin_of_run() functions etc in your derived class,
    and they will be called at the appropriate transition.
    
    See the frontend examples in the examples/ directory for more.
    
    Members:
        * client (`midas.client.MidasClient`) - The object we can use for ODB
            access etc.
        * equipment (dict of {str: `EquipmentBase`}) - Keyed by equipment name.
        * run_state (int) - The current run state (midas.STATE_RUNNING etc).
        * buffers (dict of {str: int}) - Buffer name to buffer handle. 
        * last_stats_update (datetime.datetime) - The last time we updated the
            equipment statistics in the ODB.
        * stats_update_period (datetime.timedelta) - How often we should update
            equipment statistics in the ODB.
        * _inited (bool) - Whether this base clas __init__ has been called.
    """
    def __init__(self, frontend_name):
        """
        You MUST call this function from your derived class' constructor.
        
        It creates a midas client that you can use for ODB access, and 
        registers functions that will be called on run transitions.
        
        If you need to edit your frontend name based on the frontend index
        (supplied by the -i command line flag) you should call parse_args()
        BEFORE calling this base class __init__(); the index will be available
        from frontend_index. See examples/multi_frontend.py for an example.

        Args:
            * frontend_name - The name of this client/frontend. It should be
                unique within your experiment. We do NOT do any automatic
                substitution based on the frontend index.
        """
        global cmd_line_hostname, cmd_line_exptname, args_parsed
        
        if not args_parsed:
            # Grab any host/expt provided on command-line
            parse_args()
        
        if self._is_initialized():
            logger.debug("Frontend %s is already initialized" % frontend_name)
            return

        logger.info("Initializing frontend %s" % frontend_name)
        self.client = midas.client.MidasClient(frontend_name, cmd_line_hostname, cmd_line_exptname, daemon_flag)
        
        self.equipment = {}
        self.buffers = {}
        
        self.last_stats_update = None
        self.stats_update_period = datetime.timedelta(milliseconds=3000)
        
        self.client.register_transition_callback(midas.TR_START, 500, self._tr_start_callback)
        self.client.register_transition_callback(midas.TR_STOP, 500, self._tr_stop_callback)
        self.client.register_transition_callback(midas.TR_PAUSE, 500, self._tr_pause_callback)
        self.client.register_transition_callback(midas.TR_RESUME, 500, self._tr_resume_callback)
            
        self.client.register_disconnect_callback(self._frontend_exit_callback)

        self.run_state = self.client.odb_get("/Runinfo/State")
        self.client.odb_watch("/Runinfo/State", self._run_state_callback)
        
        self._inited = True
        
    def begin_of_run(self, run_number):
        """
        You MAY implement this function in your derived class. It will be
        called at the start of each run, and you can use it to prevent the 
        run from starting if needed.
        
        Args:
            * run_number (int) - The new run number.
            
        It should return:
            * None or 1 - Transition was successful
            * int that isn't 1 - Transition failed
            * 2-tuple of (int, str) - Transition failed and the reason why
        """
        return midas.status_codes["SUCCESS"]
    
    def end_of_run(self, run_number):
        """
        You MAY implement this function in your derived class.
        
        See begin_of_run documentation.
        """
        return midas.status_codes["SUCCESS"]
    
    def pause_run(self, run_number):
        """
        You MAY implement this function in your derived class.
        
        See begin_of_run documentation.
        """
        return midas.status_codes["SUCCESS"]
    
    def resume_run(self, run_number):
        """
        You MAY implement this function in your derived class.
        
        See begin_of_run documentation.
        """
        return midas.status_codes["SUCCESS"]

    def frontend_exit(self):
        """
        You MAY implement this function in your dervived class.

        To GUARANTEE that this function gets called, you should use a context
        manager style for the frontend class (`with` syntax).

        ```
        # This style will work in most cases, but is not guaranteed to call
        # `frontend_exit()` when the program exits (depends on your specific 
        # python version/implementation and python code):
        my_fe = MyFrontend("my_fe")
        my_fe.run()
        
        # This style is guaranteed to call `frontend_exit()`:
        with MyFrontend("my_fe") as my_fe:
            my_fe.run()
        ```
        """
        pass

    def add_equipment(self, equip):
        """
        You MUST call this function to add equipment to your frontend before
        you call run(). You can call it multiple times to add several equipment
        if you want.
        
        Args:
            * equip (something that inherits from `EquipmentBase`)
        """
        if not isinstance(equip, EquipmentBase):
            raise TypeError("Equipment must inherit from EquipmentBase")
        
        if not equip._is_initialized():
            raise RuntimeError("You must call init_equip() for all Equipment")

        self.equipment[equip.name] = equip
                
    def set_equipment_status(self, equipment_name, status_text, status_color="greenLight"):
        """
        You can call this function to set the "status" of an equipment.
        The status is shown on the main midas status page.
        
        Args:
            * equipment_name (str) - The name of an equipment you added.
            * status_text (str) - This should be a short string, or else the
                formatting of the main status page may get a little wacky.
            * status_color (str) - There are 4 pre-defined colors (greenLight, 
                yellowLight, yellowGreenLight, redLight) or you can set any
                CSS background color name / hex value (max length 32 chars).
        """
        self.equipment[equipment_name].set_status(status_text, status_color)
        
    def set_all_equipment_status(self, status_text, status_color="greenLight"):
        """
        Like set_equipment_status(), but apply to same status to all equipment
        controlled by this frontend.
        """
        for equip in self.equipment.values():
            equip.set_status(status_text, status_color)
    
    def run(self):
        """
        You MUST call this function to run your frontend. 
        
        It is the main routine that talks to midas and gets events from your
        equipment as needed.
        
        It will run forever until the frontend is killed (either through a 
        signal or via the midas Programs webpage).
        """
        if len(self.equipment) is None:
            raise RuntimeError("You must add some equipment to this frontend")
        
        if not self._is_initialized():
            raise RuntimeError("You must call __init__ for this frontend")
        
        self._open_buffers()
                
        print("Running...")
        
        while True:
            should_stop_run = False
            loop_start_time = datetime.datetime.now()
            
            for equip in self._enabled_equipment():
                if equip._is_active_for_state(self.run_state):
                    if equip._is_polled() and equip.poll_func():
                        # Polled equipment says that it has data to send
                        self._get_and_send_events(equip)
                        
                    if equip._is_periodic() and equip._is_time_to_work():
                        # Periodic equipment is ready to send
                        self._get_and_send_events(equip)
                
                if equip._is_event_limit_reached():
                    logger.info("Equipment %s has reached event limit" % equip.name)
                    should_stop_run = True
                
            stats_time = datetime.datetime.now()
            if self.last_stats_update is None or stats_time - self.last_stats_update > self.stats_update_period:
                logger.debug("Time to update statitics in ODB")
                self.last_stats_update = stats_time
                
                for equip in self.equipment.values():
                    equip._update_statistics_in_odb()
                
            loop_end_time = datetime.datetime.now()
            run_time_ms = (loop_end_time - loop_start_time).total_seconds() * 1000
            
            if should_stop_run and self.run_state != midas.STATE_STOPPED:
                self.client.stop_run()
            
            self.client.lib.c_cm_check_deferred_transition()
            
            # Choose how long to yield for based on period of equipment
            # that can produce events in the current run state.
            relevant_equip = [e for e in self.equipment.values() if e._is_active_for_state(self.run_state)]
            
            if len(relevant_equip) == 0:
                yield_time = 10
            else:
                shortest_period = min([e.common["Period"] for e in relevant_equip])
                yield_time = max(0, int(shortest_period/5 - run_time_ms))
                yield_time = min(10, yield_time)
                
            self.client.communicate(yield_time)

    def _is_initialized(self):
        """
        Whether the base class __init__() has been called.
        
        Returns:
            bool
        """
        return hasattr(self, "_inited")

    def _enabled_equipment(self):
        """
        Get the equipment that are actually enabled.
        
        Returns:
            list of EquipmentBase
        """
        return [equip for equip in self.equipment.values() if equip.common["Enabled"]]

    def _get_and_send_transition_events(self, transition):
        """
        Equipment can send events at run transitions. Do that for the 
        transition currently in progress.
        
        Args:
            * transition (int) - One of midas.TR_xxx.
        """
        for equip in self._enabled_equipment():
            if equip._is_active_for_transition(transition):
                self._get_and_send_events(equip)
        
    def _get_and_send_events(self, equip):
        """
        Call _get_events() (which call's the user's readout_func()) for the
        given equipment.
        
        If an event is returned, send it to the correct midas buffer, and
        copy to the ODB if appropriate.
        
        Args:
            * equip (EquipmentBase)
        """
        events = equip._get_events()

        for event in events:
            event.populate_bank_and_event_size()
            buffer_name = equip.common["Buffer"]
            
            if buffer_name and len(buffer_name):
                buffer_handle = self._get_buffer_handle(buffer_name)
                self.client.send_event(buffer_handle, event)
                
            ev_size = event.header.event_data_size_bytes
            bank_list = ", ".join(event.banks.keys())
            logger.debug("Equipment %s has produced an event of size %s bytes with banks: %s" % (equip.name, ev_size, bank_list))
            equip._stats_add_event_of_size(ev_size)
                
            if equip._is_time_to_write_vars_to_odb():
                equip.last_odb_var_write_time = datetime.datetime.now()
                
                for bank in event.banks.values():
                    odb_path = equip.odb_variables_dir + "/" + bank.name
                    c_data = self.client._midas_type_to_ctype(bank.type, initial_value=bank.data)
                    self.client.odb_set(odb_path, c_data, explicit_new_midas_type=bank.type)
        
        if len(events) is None:
            logger.debug("Equipment %s did not produce an event" % equip.name)
            
    def _open_buffers(self):
        """
        Open and event buffers we need for our equipment.
        """
        buffer_names = set([equip.common["Buffer"] for equip in self.equipment.values()])
        
        for buffer_name in buffer_names:
            if len(buffer_name):
                self.buffers[buffer_name] = self.client.open_event_buffer(buffer_name)
        
    def _run_state_callback(self, client, path, new_value):
        """
        Callback function we register to know whether /Runinfo/State has 
        changed.
        
        Args:
            * client - Needed to match the expected interface
            * path - Needed to match the expected interface
            * new_value (int) - The new run state
        """
        self.run_state = new_value
        
    def _get_buffer_handle(self, buffer_name):
        """
        Get the buffer handle we got when we opened the specified buffer.
        
        Args:
            * buffer_name (str)
            
        Returns:
            int
        """
        return self.buffers.get(buffer_name, None)

    def _tr_start_callback(self, client, run_number):
        """
        Callback function we register for the begin-of-run transition.
        We call the begin_of_run() function that the user can override,
        and send any transition events from our equipment.
        
        Args:
            * client - Needed to match the expected interface
            * run_number (int) - The new run number
        """
        for equip in self.equipment.values():
            equip._reset_stats()
            
        # Call user's BOR function
        retval = self.begin_of_run(run_number)
        
        if retval == midas.status_codes["SUCCESS"]:
            self._get_and_send_transition_events(midas.TR_START)
            
        return retval
        
    def _tr_stop_callback(self, client, run_number):
        """
        Callback function we register for the end-of-run transition.
        We call the end_of_run() function that the user can override,
        and send any transition events from our equipment.
        
        Args:
            * client - Needed to match the expected interface
            * run_number (int) - The current run number
        """
        retval = self.end_of_run(run_number)
        
        if retval == midas.status_codes["SUCCESS"]:
            self._get_and_send_transition_events(midas.TR_STOP)
            
        self.client.lib.c_rpc_flush_event();
        for buf in self.buffers.values():
            self.client.lib.c_bm_flush_cache(buf, midas.BM_WAIT);

        return retval
    
    def _tr_pause_callback(self, client, run_number):
        """
        Callback function we register for the pause run transition.
        We call the pause_run() function that the user can override,
        and send any transition events from our equipment.
        
        Args:
            * client - Needed to match the expected interface
            * run_number (int) - The new run number
        """
        retval = self.pause_run(run_number)
        
        if retval == midas.status_codes["SUCCESS"]:
            self._get_and_send_transition_events(midas.TR_PAUSE)
            
        return retval
    
    def _tr_resume_callback(self, client, run_number):
        """
        Callback function we register for the resume run transition.
        We call the resume_run() function that the user can override,
        and send any transition events from our equipment.
        
        Args:
            * client - Needed to match the expected interface
            * run_number (int) - The new run number
        """
        retval = self.resume_run(run_number)

        if retval == midas.status_codes["SUCCESS"]:
            self._get_and_send_transition_events(midas.TR_RESUME)
            
        return retval
        
    def _frontend_exit_callback(self, client):
        """
        Callback function we register to be called just before we disconnect 
        from midas. We call the `frontend_exit()` function that the user can 
        override.
        
        Args:
            * client - Needed to match the expected interface
        """
        self.frontend_exit()

    def __enter__(self):
        """
        Context manager (`with` style).
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Context manager (`with` style).
        """
        # Will call `frontend_exit()`.
        self.client.disconnect()