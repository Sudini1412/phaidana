import midas.client
import threading
import time
import queue
import datetime
import math
import copy
import traceback
import collections
import ctypes
import os.path
import argparse
import logging
import numbers
import sys
import shutil

try:
    import numpy as np
    have_numpy = True
except ImportError:
    have_numpy = False

logger = logging.getLogger('sequencer')

# TODO - May want tracing in other modules (not just the main script). Add a function "seq.trace_module()"
#           that user can call to request tracing in certain files. Need to match logic of MSL
#           sequencer for updating ODB state/lines etc.

class SequenceClient(midas.client.MidasClient):
    """
    This class contains all the midas-related functions that you can use as part of
    your sequencer script. Unlike a regular midas client, your script does NOT need
    to periodically call `communicate()`, even if it's going to be running for a 
    really long time.

    Your script has access to a global variable named `seq`, which is an instance of
    this class. The `seq` object is also passed to the three main functions you can
    define (`define_params()`, `sequence()` and `at_exit()`).

    As well as the functions defined here, you have access to all of the
    functions in `midas.client.MidasClient`, such as `odb_set()` and `odb_get()`.

    An example sequencer script is:

    ```
    # You can import arbitrary python modules
    import time

    # You can define/call arbitrary functions
    def subfn(a):
        b = a*6
        return a^b

    # You MAY define this special function, which will get run when we
    # first load the script. It should call `register_param()` for any
    # "parameters" that you want the user to be able to set when they are
    # starting the sequence.
    def define_params(seq):
        seq.register_param("num_loops", "Number of loops", 2)
        seq.register_param("do_sleep", "Do extra sleep", True)

    # You MUST define this special function, which will get run when the user
    # starts the sequence. Progress through the script is displayed on a webpage.
    def sequence(seq):
        # You can call arbitrary functions that are defined either in your script
        # or in external modules. We only track progress through functions in the
        # main script, not those in other modules. i.e. we'll show the line-by-line
        # progress through `subfn()`, but will only show that we're in `time.sleep()`,
        # not whatever the internal logic of `time.sleep()` is.
        subfn()
        time.sleep(3.5)

        # You can access the value of "parameters" you defined
        num_loops = seq.get_param("num_loops")

        # If you use `seq.range()` instead of the regular `range()`, the webpage will
        # show a progress bar of how far through the loop we are. Nested loops are
        # supported up to 10 levels deep (you can have deeper nesting, we just won't show
        # the progress on the webpage).
        for i in seq.range(num_loops):
            print(i)

            # If you use `seq.wait_seconds()` instead of `time.sleep()`, we'll track the
            # progress of the sleep on the webpage using a progress bar.
            seq.wait_seconds(1.5)

        # You can use most functions from `midas.client.MidasClient`
        rn = seq.odb_get("/Run info/Run number")

        if seq.get_param("do_sleep"):
            print("Doing extra sleep")
            seq.wait_seconds(3)

        # You can start and stop other programs, and wait until they're running.
        seq.start_client("Logger", True)
        seq.wait_clients_running(["Logger"])

        # You can start and stop runs
        seq.start_run()

        # You can send messages that will be displayed on the webpage
        seq.sequencer_msg("We started a run!")

        seq.wait_seconds(10)
        seq.stop_run()

        print("fin")

    # You MAY define this special function that will get run when the main
    # `sequence()` function finishes (even if the main function raised an exception).
    def at_exit(seq):
        if seq.odb_get("/Runinfo/State") != midas.STATE_STOPPED:
            seq.stop_run()
    ```
    """
    def __init__(self, client_name, host_name, expt_name, debug_flag, odb_base):
        # Most param passed to `midas.client.MidasClient()` constructor.
        # `self.odb_base` is needed as you can potentially run multiple
        # sequencers in parallel (at `/PySequencer`, `/PySequencerXXX` etc in the ODB).
        try:
            super().__init__(client_name, host_name, expt_name, debug_flag, throw_if_already_running=True)
        except RuntimeError:
            print(f"{client_name} is already running")
            exit(1)

        self.odb_base = odb_base
        
        # If we're logging to midas message log (rather than just stdout), tell the
        # logging handler about which MidasClient it can use to call msg().
        for handler in logger.handlers:
            if isinstance(handler, midas.MidasLogHandler):
                handler.client = self

        logger.info(f"Starting PySequencer with base ODB directory of {self.odb_base}")

    def register_param(self, name, comment, default_value, options=[]):
        """
        Only call this from your `define_params()` function!

        This will register a parameter that the user will be asked to set
        when they start the sequence.

        Args:
            * name (str) - parameter name, must be < 32 chars long
            * comment (str) - an explanation of what the parameter is for
            * default_value (int/float/bool/str etc) - default value for when
                the parameter is first created. Also used for determining the
                correct type to use in the ODB.
            * options (list of str) - If you want the user to choose from a set 
                of options in a drop-down menu (rather than being able to type
                arbitrary values), you can set those here. Probably only works
                correctly if `default_value` is a string!
        """
        # Differences to MSL sequencer. We report all local/global vars in /Variables.
        # We don't copy all /Param/Value to /Variables.
        # We allow arbitrary types for values/variables, not just strings.
        comms.state_queue.put(ParamInfo(name, comment, default_value, options))

    def get_param(self, name):
        """
        Get the value a user set for a given parameter.

        Args:
            * name (str)

        Returns:
            int/float/bool/str etc, whatever is in the ODB
        """
        path = f"{self.odb_base}/Param/Value/{name}"

        if not self.odb_exists(path):
            path = f"{self.odb_base}/Param/Defaults/{name}"
        
        if not self.odb_exists(path):
            raise ValueError(f"Parameter '{name}' does not exist! Fix your script to call `seq.register_param()` in your `define_params()` function!")
        
        return self.odb_get(path)

    def sequencer_msg(self, text, wait=False):
        """
        Send a message to display on the sequencer webpage, and optionally wait until the 
        user has acknowledged it before continuing.

        Note this is different to the `midas.client.MidasClient.msg()` function that just
        writes a message to the midas message log.

        Args:
            * text (str) - the message to display
            * wait (bool) - whether to wait for the user to acknowledge the message before
                continuing the sequencer script.
        """
        comms.message_acknowledged = False
        comms.state_queue.put(MessageInfo(text, wait))

        if wait:
            while True:
                time.sleep(0.1)
                
                if comms.message_acknowledged or comms.stop:
                    break
                
    def set_run_description(self, desc):
        """
        Set the ODB key "/Experiment/Run Parameters/Run Description" to the given value.

        Args:
            * desc (str) - the description to set
        """
        self.odb_set("/Experiment/Run Parameters/Run Description", desc)

    def set_py_logger_debug(self, debug):
        """
        Same as `set_py_logger()`, but only affecting the `debug` argument.
        """
        self.set_py_logger(debug, None)

    def set_py_logger_to_midas(self, to_midas_msg_log):
        """
        Same as `set_py_logger()`, but only affecting the `to_midas_msg_log` 
        argument.
        """
        self.set_py_logger(None, to_midas_msg_log)

    def set_py_logger(self, debug, to_midas_msg_log):
        """
        Configure python logging - i.e. what happens when `logger.info("xyz")` etc.
        are called.

        Note that this can also be set using the command-line flags
        `--verbose` and `--log-to-midas`.

        Args:
            * debug (bool) - if True, `logger.debug()` statements will be handled; 
                if False they will be ignored (and only info/warning/error messages
                will be handled); if None, no changes will be made.
            * to_midas_msg_log (bool) - if True, statements will be copied to the 
                midas message log; if False, they will just be printed to screen;
                if None, no changes will be made.
        
        Summary of what happens for different output:
            * `print()` – always printed to screen; never written to midas log.
            * `logger.info()` / `logger.error()` – always printed to screen; 
                written to midas log if `to_midas_msg_log` is True.
            * `logger.debug()` – printed to screen if `debug` is True; 
                written to midas log if both `debug` and `to_midas_msg_log` are True.
        """
        if debug is not None:
            logger.setLevel(logging.DEBUG if debug else logging.INFO)

        if to_midas_msg_log is not None:
            curr_is_midas = False

            for handler in logger.handlers:
                if isinstance(handler, midas.MidasLogHandler):
                    curr_is_midas = True

            if to_midas_msg_log and not curr_is_midas:
                # Change from print to midas
                logger.handlers.clear()
                logger.addHandler(midas.MidasLogHandler(self, "pysequencer"))

            if curr_is_midas and not to_midas_msg_log:
                # Change from midas to print
                logger.handlers.clear()
                logger.addHandler(logging.StreamHandler())

    def wait_func(self, func, check_period_secs=0.1):
        """
        Wait until an arbitrary condition is met. 
        
        The function you specify should return a boolean - True if the 
        sequencer should continue, or False if we should wait and try again.

        If you don't want to write a separate function, you can specify a lambda
        as the function to call. See examples below for the syntax.

        Args:
            * func (function/lambda) - code to run
            * check_period_secs (float) - how often to check the condition

        Example usage:

        ```
        # Using a lambda (anonymous one-line function)
        def sequence(seq):
            seq.wait_func(lambda : seq.odb_get("/path/aaa") > 1.5 and seq.odb_get("/path/bbb") < 3.11)
        ```

        ```
        # Using any function you want, so long as it returns a boolean
        import random

        def custom_func():
            val = random.randint(0, 10)
            print(f"random val was {val}")

            return val == 7

        def sequence(seq):
            # DO NOT WRITE ANY PARENTHESES - just state the name of the function
            # that we should call to check if the condition is met or not.
            seq.wait_func(custom_func)
        ```
        """
        # Implementation detail - we do not allow the user to pass a string
        # that we eval() or exec(), as the SequenceClient object is created
        # in the main thread (and before we exec() the user's sequence script), 
        # and sees a different environment!
        # So if the user defines a function "custom_func()" in their script,
        # they can call it from their code, but we cannot call it from within
        # the SequenceClient object. It's an odd edge-case that is best avoided
        # by only allowing the user to pass in functions / lambdas to `wait_func()`.
        if not callable(func):
            raise TypeError("Pass a callable function or lambda")

        while True:
            curr_value = func()

            if curr_value:
                break
            
            comms.state_queue.put(WaitInfo("Function", curr_value, 1))

            time.sleep(0.1)

        comms.state_queue.put(WaitInfo("", 0, 0))

    def wait_odb(self, path, op, target, between_upper_target=None, stable_for_n_secs=None):
        """
        Wait until a certain condition is met in the ODB.

        E.g. `wait_odb("/Path/to/something", ">", 6.5)` will wait until the ODB value
        "/Path/to/something" exceeds 6.5 before continuing the sequencer script.

        Args:
            * path (str) - ODB path
            * op (str) - one of "<", ">", "<=", ">=", "==", "!=", "between"
            * target (int/float/str etc) - the value to compare to
            * between_upper_target (int/float/None) - if `op` is "between", we'll check if
                the value is >= `target` and <= `between_upper_target`. Ignored if `op`
                is not "berween"
            stable_for_n_secs (int/float/None) - require that the condition is met for at least
                N seconds (i.e. ensuring that the condition is stable, not just transient).
                If None, we'll return as soon as the condition is met for the first time.
        """
        if op not in ["<", ">", "<=", ">=", "==", "!=", "between"]:
            raise ValueError(f"Unsupported operation '{op}' passed to wait_odb()")
        
        if not self.odb_exists(path):
            raise ValueError(f"Non-existant ODB path '{path}' passed to wait_odb()")

        if op == "between" and between_upper_target is None:
            raise ValueError("Must supply both 'target' and 'between_upper_target' when op is \"between\"")

        stable_start_time = None

        while True:
            curr_value = self.odb_get(path)
            valid = False
            
            if op == "<" and curr_value < target:
                valid = True
            elif op == ">" and curr_value > target:
                valid = True
            elif op == "<=" and curr_value <= target:
                valid = True
            elif op == ">=" and curr_value >= target:
                valid = True
            elif op == "==" and curr_value == target:
                valid = True
            elif op == "!=" and curr_value != target:
                valid = True
            elif op == "between" and curr_value >= target and curr_value <= between_upper_target:
                valid = True

            if valid:
                if stable_for_n_secs is None:
                    # We're done as soon as the condition is met
                    break
                
                now = datetime.datetime.now()

                if stable_start_time is None:
                    # We're just starting our validity period
                    stable_start_time = now

                delta_secs = (now - stable_start_time).total_seconds()
                
                if delta_secs > stable_for_n_secs:
                    # We've been stable for long enough
                    break

                started_ms = int(stable_start_time.timestamp() * 1000)
                comms.state_queue.put(WaitInfo("ODBValueStability", delta_secs, stable_for_n_secs, odb_path=path, started_ms=started_ms))
            else:
                # We're not valid - need to reset the clock if we're checking for stability
                stable_start_time = None
                comms.state_queue.put(WaitInfo("ODBValue", curr_value, target, odb_path=path))

            time.sleep(0.1)

        comms.state_queue.put(WaitInfo("", 0, 0))

    def sleep(self, value):
        """
        Alias for `wait_seconds()` as it feels so much more natural to write
        `seq.sleep()` instead of `seq.wait_seconds()`! But `wait_seconds()`
        matches the rest of the `wait_xxx()` functions, so let the user decide
        which they prefer!
        """
        self.wait_seconds(value)

    def wait_seconds(self, value):
        """
        Wait until a certain amount of time has elapsed before continuing the sequencer script.

        Args:
            * value (float/int) - time to wait in seconds
        """
        start = datetime.datetime.now()
        remaining = value
        start_int = int(start.timestamp() * 1000)

        while True:
            now = datetime.datetime.now()
            elapsed = (now - start).total_seconds()
            remaining = value - elapsed
            comms.state_queue.put(WaitInfo("Seconds", elapsed, value, started_ms=start_int))

            if remaining <= 0:
                break

            time.sleep(min(remaining, 0.1))

        comms.state_queue.put(WaitInfo("", 0, 0))

    def wait_events(self, target):
        """
        Wait until the ODB value "/Equipment/Trigger/Statistics/Events sent" exceeds
        a certain value before continuing the sequencer script.
        
        Note that this is implemented to match the functionality of the MSL-based sequencer, 
        but it is unlikley to be useful for most experiments (unless you have an equipment 
        named "Trigger"!).

        Args:
            * target (int) - how many events to wait for
        """
        while True:
            curr_value = self.odb_get("/Equipment/Trigger/Statistics/Events sent")
            comms.state_queue.put(WaitInfo("Events", curr_value, target))

            if curr_value >= target:
                break

        comms.state_queue.put(WaitInfo("", 0, 0))

    def wait_clients_running(self, client_names, timeout_secs=10):
        """
        Wait until a certain set of clients are running before continuing the sequencer script.

        Useful if you need to start/stop programs as part of your sequencer script (using 
        `start_client()`, `start_all_required_programs()` etc.). You should try to start them, then
        call this function to make sure they're running before continuing the script.

        If the clients do not start within the specified timeout, execution of the sequencer
        script will stop.

        Args:
            * client_names (str or list of str) - client/program names to check
            * timeout_secs (float/int) - how long to wait before giving up and concluding
                that the client/program isn't running or failed to start successfully when
                you previously called `start_client()` etc.
        """
        if not isinstance(client_names, list):
            client_names = [client_names]

        start = datetime.datetime.now()
        remaining = timeout_secs
        start_int = int(start.timestamp() * 1000)
        missing_clients = []

        while True:
            now = datetime.datetime.now()
            elapsed = (now - start).total_seconds()
            remaining = timeout_secs - elapsed
            comms.state_queue.put(WaitInfo("ClientsRunning", elapsed, timeout_secs, started_ms=start_int))

            clients_status = self.clients_exist(client_names, return_as_dict=True)

            missing_clients = [name for name,running in clients_status.items() if not running]

            if len(missing_clients) == 0:
                break

            if remaining <= 0:
                break

            time.sleep(min(remaining, 0.1))

        comms.state_queue.put(WaitInfo("", 0, 0))

        if len(missing_clients) > 0:
            raise RuntimeError(f"Timeout waiting for programs {missing_clients} to be running")

    def disable_tracing(self):
        """
        Disable tracing/tracking of the current sequence state. You might want to do
        this if you have a "hot" loop that gets executed many times. 
        
        Tracing each iteration of such a loop may slow down your program by a factor 2 
        or more. Note that this also applies to "hidden" loops like list comprehensions.

        Only use this functionality if you've identified a hot loop that is affecting 
        performance of your script.

        ```
        # The webpage will just show the state at `do_a()` and `do_b()`, skipping all
        # updates that we're in `do_something_quick()`. May speed up the program a lot.
        do_a()

        seq.disable_tracing()

        for i in range(1000000):
            do_something_quick()

        seq.enable_tracing()

        do_b()
        ```
        """
        global _user_tracing_enabled
        _user_tracing_enabled = False

    def enable_tracing(self):
        """
        Opposite of `enable_tracing()`.
        """
        global _user_tracing_enabled
        _user_tracing_enabled = True

    class range:
        """
        Use this in a for loop to track progress on the Sequencer webpage.

        Argument usage matches the standard python `range()` function:
            * `seq.range(4)` -> [0, 1, 2, 3]
            * `seq.range(2, 6)` -> [2, 3, 4, 5]
            * `seq.range(20, 50, 10)` -> [20, 30, 40]

        Full usage would be like:

        ```
        for i in seq.range(5):
            seq.odb_set("/path/to/something", i)
            seq.wait_seconds(2.5)
        ```

        Each time we iterate over the loop, a progress bar on the sequencer webpage will be updated.
        """
        def __init__(self, val1, val2=None, step=1):
            if val2 is None:
                self.start = 0
                self.end = val1
            else:
                self.start = val1
                self.end = val2

            self.step = step

            if not isinstance(self.start, int):
                raise TypeError("start must be an integer")
            if not isinstance(self.end, int):
                raise TypeError("end must be an integer")
            if not isinstance(self.step, int):
                raise TypeError("step must be an integer")
            
            self.curr_val = self.start
            self.n_steps = math.floor((self.end - self.start) / self.step)
            self.curr_step = 0
            self.add_state_to_queue("init")

        def __iter__(self):
            try:
                # Yield each value as needed
                while self.curr_step < self.n_steps:
                    val = self.start + self.curr_step * self.step
                    self.add_state_to_queue("step")
                    yield val
                    self.curr_step += 1

                # We only reach here once we've yielded all the values the user wanted
                self.add_state_to_queue("complete")
            except GeneratorExit:
                # Something happened so we exited early (e.g. user called break)
                self.add_state_to_queue("early_exit")

        def add_state_to_queue(self, what):
            comms.state_queue.put(LoopInfo(id(self), what, self.n_steps, self.curr_step))

    # There are several functions in `midas.client.MidasClient()` that would
    # be problematic to use in a sequencer script, as they use callback functions
    # that get called from `client.communicate()`. In PySequencer, the main
    # thread calls communicate(), not the other thread where the script is running.
    # So we would lose the "linearity" of the script logic. And there would be
    # contention between whether the main thread or spawned thread ran the callback.
    #
    # If we wanted scripts to be able to use callbacks, we would have to run the script in a 
    # subprocess (instead of a thread) and have the user call `communicate()` very
    # often in their script, which is not very ergonomic.
    # We therefore restrict all of these problematic functions to being called from the
    # main thread only (so that the PySequencer itself can still use `odb_watch()`, even
    # if the user's script is prohibited from doing so).

    def communicate(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().communicate(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("communicate")
        
    def odb_watch(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().odb_watch(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("odb_watch")
        
    def register_transition_callback(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().register_transition_callback(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("register_transition_callback")
    
    def register_brpc_callback(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().register_brpc_callback(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("register_brpc_callback")
    
    def register_jrpc_callback(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().register_jrpc_callback(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("register_jrpc_callback")
        
    def register_disconnect_callback(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().register_disconnect_callback(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("register_disconnect_callback")
        
    def register_message_callback(self, *args, **kwargs):
        # Only allow the main thread to call this function, not the thread that's running
        # the user's script.
        if threading.current_thread() is threading.main_thread():
            super().register_message_callback(*args, **kwargs)
        else:
            raise InvalidSequencerCommand("register_message_callback")
        
#
# Most users should not need to know about the functions below here, as they are
# related to the internal running and tracking of the sequencer script.
#

"""
Implementation details for PySequencer developers.

There are 2 threads. The main thread handles user interactions, and
updating the state in the ODB etc. The second threads runs the user's
script, and uses debugging tools to track which line we're currently on
in the script, and the state of local variables etc. The threads communicate
using standard python multithreading Events and Queues, which are encapsulated
in the global `comms` object.

The user's code runs in its own global environment and cannot stomp on
either of our threads unless it tries really really hard.

We slightly abuse python's settrace() functionality to not only trace which
lines are being executed in the script, but also to implement various delays
while we're waiting for action from the user (e.g. when waiting for them to 
acknowledge a message before continuing the script, we loop in our settrace()
callback until we see that the relevant Event has been set by the main thread).
It does mean that if the user calls a very slow function from a different
module, we may not be able to actually stop execution of the script until that
function returns. If the user calls a blocking function, we may not be able
to stop execution at all except by killing the overall PySequencer client and
starting over.  
"""

class ThreadComms:
    """
    Wrapper object for inter-thread communication.

    Members for main thread -> sequence thread communication:
        * pause - user has asked to pause the script, and we
            won't move to the next line until they unpause
        * stop - user has asked to stop the script
        * debug - user is in debug mode and we shouldn't move to
            the next line in the script until they say "step over:
        * stop_over - user is in debug mode and wants to move to
            the next line in the script
        * message_acknowledged - the script called `sequencer_msg()`
            with "wait=True", and the user has now acknowledged the
            message, so we can move to the next line in the script

    Members for sequence thread -> main thread communication:
        * state_queue - updates of what the sequence thread is doing
            (e.g. what line it is on, if there has been an exception,
            progress through loops etc). All updates are one of the
            xxxInfo classes defined below.
    """
    def __init__(self):
        self.pause = False
        self.stop = False
        self.debug = False
        self.step_over = False
        self.message_acknowledged = False
        self.state_queue = queue.Queue()

    def reset(self):
        """
        Reset the inter-thread communication state (except self.debug,
        which is really a "level" rather than a flag that the sequencer
        thread clears).
        """
        self.pause = False
        self.stop = False
        self.step_over = False
        self.message_acknowledged = False

        while not self.state_queue.empty():
            self.state_queue.get()

comms = ThreadComms()

class StopSequencerException(Exception):
    """
    Trivial exception for indicating that the sequencer thread exited because 
    `stop` was set.
    """
    pass

class InvalidSequencerCommand(Exception):
    """
    Trivial exception for telling the user that their script contained a command
    that is verboten.
    """
    def __init__(self, cmd):
        message = f"You cannot use callback-based functions (such as {cmd}) in Sequencer scripts"
        super().__init__(message)

class WaitInfo:
    """
    Used by the sequence thread to tell the main thread that we're waiting for
    some condition to be met.

    Members:
        * wait_type (str) - one of "ODBValue", "ODBValueStability", "Seconds", "Events", "ClientsRunning", "Func"
        * curr_value (float/int/str etc) - only int/float/bool get correctly reported in ODB
        * target_value (float/int/str etc) - only int/float/bool get correctly reported in ODB
        * started_ms (int) - UNIX timestamp if `wait_type` is "Seconds", "ClientsRunning" or "ODBValueStability"
        * odb_path (str) - path we're watching if `wait_type` is "ODBValue" or "ODBValueStability"
    """
    def __init__(self, wait_type, curr_value, target_value, started_ms=0, odb_path=""):
        self.wait_type = wait_type
        self.curr_value = curr_value
        self.target_value = target_value
        self.started_ms = started_ms
        self.odb_path = odb_path

class ScriptInfo:
    """
    Used by the sequence thread to tell the main thread what the code in the script is.

    Members:
        * lines (list of str) - lines of code in user's script
    """
    def __init__(self, lines):
        self.lines = lines

class MessageInfo:
    """
    Used by the sequence thread to tell the main thread that we should write a message in
    the ODB, which will be displayed by the sequencer webpage.

    Members:
        * text (str) - message to display
        * wait (bool) - whether we should wait for the user to acknowledge the message
            before continuing execution of the script
    """
    def __init__(self, text, wait):
        self.text = text
        self.wait = wait

class ResetParamsInfo:
    """
    Used by the sequence thread to tell the main thread that we should clear the old
    list of Params in the ODB.
    """
    pass

class FinishedParamsInfo:
    """
    Used by the sequence thread to tell the main thread that we've finished the user's
    define_params() function.
    """
    pass

class ScriptLoadedInfo:
    """
    Used by the sequence thread to tell the main thread that we've finished loading the
    user's script and can reset the "Load new file" flag in the ODB.
    """
    pass

class ParamInfo:
    """
    Used by the sequence thread to tell the main thread that the user wants to register a
    parameter.

    Members:
        * name (str)
        * comment (str)
        * default_value (int/float/str/bool etc)
        * options (list of str)
    """
    def __init__(self, name, comment, default_value, options):
        self.name = name
        self.comment = comment
        self.default_value = default_value
        self.options = options

class StateInfo:
    """
    Used by the sequence thread to tell the main thread about the current execution 
    state of the user's script.

    Members:
        * filename (str) - filename we're currently in. Will be "<string>" for the
            user's script (as we read the file then execute it via exec()). Will be
            a real filename for any other modules that the user calls.
        * line_num (int) - current line of code we're on in the file
        * func_name (str) - name of the function we're currently in
        * these_locals (dict) - local variables for the current context
        * these_globals (dict) - global variables for the current context
    """
    def __init__(self, filename, line_num, func_name, these_locals, these_globals):
        self.filename = filename
        self.line_num = line_num
        self.func_name = func_name
        self.variables = {}

        for k, v in these_locals.items():
            self.variables[k] = v

        for k, v in these_globals.items():
            if k.startswith("__"):
                continue

            self.variables[k] = v

class LoopInfo:
    """
    Used by the sequence thread to tell the main thread about progress through a
    loop that is being controlled by `SequenceClient.range()`.

    Members:
        * id (int) - identity of the current `range()` object
        * what (str) - one of "init", "step", "complete", "early_exit"
        * n (int) - total number of steps expected
        * counter (int) - current step number
    """
    def __init__(self, id, what, n, counter):
        self.id = id
        self.what = copy.copy(what)
        self.n = copy.copy(n)
        self.counter = copy.copy(counter)

class ExceptionInfo:
    """
    Used by the sequence thread to tell the main thread about exceptions
    that happened.

    Members:
        * when (str) - one of "load", "exec", "define_params", "sequence", "at_exit"
        * e (Exception) - the exception that was raised
    """
    def __init__(self, when, e):
        self.when = when
        self.e = e

_user_tracing_enabled = True

def should_trace(filename, func_name):
    # We only trace the top-level calls from the code we've loaded, and the code in
    # sequencer/client that we might need to introspect.
    if filename == "<string>":
        return _user_tracing_enabled
    if filename == "client.py":
        return func_name in ["start_run", "stop_run", "pause_run", "resume_run", "_run_transition"]
    return False

def trace_lines(frame, event, arg):
    """
    Callback that we use as part of python's settrace logic.

    This one traces each line within a function that we care about, and
    uses a queue to tell the main thread where we are.

    It also implements the debugging/stepping over logic so the user
    can step through each line in their script one-by-one. The main 
    thread sets events to say what the user has requested.
    """
    if event != "line":
        return

    if threading.current_thread() is threading.main_thread():
        return

    func_name = frame.f_code.co_name
    
    if func_name == "<module>" or func_name == "<lambda>":
        # Just entering the module or running a lambda. 
        # No need to trace these.
        return
    
    filename = os.path.split(frame.f_code.co_filename)[-1]

    if not should_trace(filename, func_name):
        return
    
    line_num = frame.f_lineno
    these_locals = frame.f_locals
    these_globals = frame.f_globals

    comms.state_queue.put(StateInfo(filename, line_num, func_name, these_locals, these_globals))

    if filename == "<string>":
        logger.debug(f"Line: {filename}: {line_num}")

        # Only handle debugging/stepping at the main script level
        if comms.debug:
            while comms.debug and not comms.step_over and not comms.stop:
                time.sleep(0.1)

        if comms.step_over:
            comms.step_over = False

def trace_calls(frame, event, arg):
    """
    Callback that we use as part of python's settrace logic.

    This one traces each function call, and sets up line-by-line
    tracing if we're in the user's script.

    It also implements pausing/stopping the script, based on events
    set by the main thread.
    """
    if event != "call":
        return None
    
    if threading.current_thread() is threading.main_thread():
        return None
    
    filename = os.path.split(frame.f_code.co_filename)[-1]
    func_name = frame.f_code.co_name

    # Allow pausing/stopping even if in other functions
    if comms.pause:
        logger.info("User has paused the script")

        while comms.pause and not comms.stop:
            time.sleep(0.1)

        logger.info("Resuming script")

    if comms.stop:
        logger.info("User has stopped the script")
        raise StopSequencerException()
    
    if should_trace(filename, func_name):
        if func_name != "__init__":
            logger.debug(f"Call: {filename}: {func_name}")
        return trace_lines
    
    return None

class ScriptExecutionThread(threading.Thread):
    """
    This thread is responsible for loading and executing the user's script.

    General concept - we run the user's code in a separate thread,
    and also use a separate "globals" environment so they can't
    accidentally stomp on any of the main thread's code. The only shared
    resource is the `SequenceClient` object, which we expose to the user's
    code as the global `seq` variable.

    Members:
        * seq (`SequenceClient`)
        * filename_to_load (str or None) - if a str, the full path to a .py script
            to load. If None, we'll use `code_lines` instead.
        * only_load (bool) - if True, we're only loading the script and running
            `define_params()`. If False we're also running `sequence()` and `at_exit()`.
        * keep_modules (list of str) - modules that should not be touched when we
            try to reload any that the user is using in their script.
        * code_lines (list of str) - if `filename_to_load` is None, we'll use this
            code instead.
        * globals_for_exec (dict) - environment for running the user's script.
    """
    
    def set_args(self, seq, filename_to_load, only_load, keep_modules, code_lines=[]):
        self.seq = seq
        self.filename_to_load = filename_to_load
        self.only_load = only_load
        self.keep_modules = keep_modules
        self.code_lines = code_lines
        self.globals_for_exec = {}

    def load(self):
        """
        Load the user's script and try to run their `define_params()` function
        if it exists.

        Failures are reported to the main thread via the state queue.

        Returns:
            bool - whether loading of the script was successful
        """
        if self.filename_to_load is None:
            logger.info(f"Reloading script from ODB ...")
        else:
            logger.info(f"Loading script {self.filename_to_load} ...")

        self.globals_for_exec = {"seq": self.seq, "logger": logger}

        # Unload any modules that were previously loaded (except for ones 
        # that the main script needs).
        #
        # Touching `sys.modules`` is normally a really bad way of doing this,
        # but it's okay in this specific situation as the only code that
        # cares about these modules is running via exec() and has its own
        # globals.
        #
        # Doing it this way is also simpler than parsing the user's code
        # to look for "import x" and "from x import y" lines and trying to
        # automatically use importlib.reload(), which would come with lots
        # of subtleties. It's also more user-friendly than having the user
        # call importlib.reload() themselves.
        for mod in list(sys.modules.keys()):
            if mod not in self.keep_modules:
                logger.debug(f"Deleting module {mod}")
                del sys.modules[mod]

        if self.filename_to_load is None:
            # Load from ODB
            code = "\n".join(self.code_lines)
        else:
            try:
                with open(self.filename_to_load) as f:
                    code = f.read()
            except Exception as e:
                comms.state_queue.put(ExceptionInfo("load", e))
                return False
        
        comms.state_queue.put(ScriptInfo(code.splitlines()))

        try:
            # Parse the user's code
            exec(code, self.globals_for_exec)
        except Exception as e:
            comms.state_queue.put(ExceptionInfo("exec", e))
            return False
        
        comms.state_queue.put(ResetParamsInfo())

        if "define_params" in self.globals_for_exec:
            logger.info("Running user's `define_params()` function")

            try:
                exec("define_params(seq)", self.globals_for_exec)
            except Exception as e:
                comms.state_queue.put(ExceptionInfo("define_params", e))
                return False
        else:
            logger.info("This script doesn't contain a `define_params()` function, so no params needed")

        comms.state_queue.put(FinishedParamsInfo())

        return True

    def run(self):
        """
        Load the user's script and try to run their `define_params()`, `sequence()` and
        `at_exit()` functions (if they exist). If `self.only_load` is True, we skip the
        last 2 items.

        Failures are reported to the main thread via the state queue.

        Note that this `run()` function is called automatically when the main thread
        calls `start()` on this thread!
        """
        loaded = self.load()
        
        comms.state_queue.put(ScriptLoadedInfo())
        
        if not loaded:
            # Failed to load script
            return
        
        logger.info("Script loaded successfully")

        if self.only_load:
            return

        logger.info("Running user's `sequence()` function...")

        if "sequence" in self.globals_for_exec:
            try:
                exec("sequence(seq)", self.globals_for_exec)
            except Exception as e:
                comms.state_queue.put(ExceptionInfo("sequence", e))
                # Don't return here, as we want to try running at_exit(), even
                # if the main sequence() aborted early.
        else:
            err = NotImplementedError("You need to provide a function called `sequence` in your script!")
            comms.state_queue.put(ExceptionInfo("sequence", err))
            return
        
        logger.info("Finished running sequence")
        
        if "at_exit" in self.globals_for_exec:
            logger.info("Running user's `at_exit()` function...")

            try:
                exec("at_exit(seq)", self.globals_for_exec)
            except Exception as e:
                comms.state_queue.put(ExceptionInfo("at_exit", e))
                return
        
            logger.info("Finished running at_exit")

class SequenceRunner:
    """
    This class contains the "main thread" code for:
        * interacting with the /PySequencer area in the ODB (responding to user 
            commands and reporting the sequencer state)
        * spawning a separate thread for loading/executing the user's script
            (`ScriptExecutionThread`)
        * telling the other thread about user's commands
        * reading messages from the other thread and updating the ODB state

    Members:
        * odb_update_period (`datetime.timedelta`) - how often to update state in ODB
        * odb_update_last (`datetime.datetime`) - when we last updated the state in ODB
        * latest_line_state (`StateInfo`) - state to write to ODB
        * latest_wait_info (`WaitInfo`) - state to write to ODB
        * loop_states (list of `LoopInfo`) - state to write to ODB
        * max_loop_odb (int) - max nesting of loops we can show in ODB
        * client_name (str) - "PySequencer", with an optional suffix if running multiple instances
        * odb_base (str) - base location in ODB for this instance
        * seq (`SequenceClient`)
        * stop_after_run (bool) - user has requested we stop sequence after this run
        * start_script (bool) - user has issued "start script" command
        * load_new_file (str or None) - if a str, user has requested we load this file
        * exec_thread (`ScriptExecutionThread` or None)
        * keep_modules (list of str)
    """
    def __init__(self, args):
        self.odb_update_period = datetime.timedelta(seconds=1)
        self.odb_update_last = datetime.datetime.now()
        self.latest_line_state = None
        self.latest_wait_info = None
        self.loop_states = []
        self.max_loop_odb = 10
        self.client_name = f"PySequencer{args.c}"
        self.odb_base = f"/{self.client_name}"
        self.stop_after_run = False
        self.start_script = False
        self.load_new_file = None
        self.exec_thread = None
        self.keep_modules = list(sys.modules.keys())
        self.params = {"Value": {}, "Defaults": {}, "Comment": {}, "Options": {}}

        daemon_flag = None

        if args.D and args.O:
            raise ValueError("Only -D or -O may be specified, not both")
        elif args.D:
            daemon_flag = 0
        elif args.O:
            daemon_flag = 1

        self.seq = SequenceClient(self.client_name, args.h, args.e, daemon_flag, self.odb_base)

        if not self.seq.odb_exists(self.odb_base):
            # Try to copy example script to correct destination if this is first time
            # pysequencer has been run.
            src = os.path.join(os.environ["MIDASSYS"], "python", "examples", "pysequencer_script_basic.py")
            dst = self.get_full_script_file_path("pysequencer_script_basic.py")

            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                logger.info(f"Initialised example script in {dst}")
            except Exception as e:
                logger.warning(f"Failed to copy example script to {dst}. Please copy it manually.")

        default_commands = collections.OrderedDict([
            ("Start script", False),
            ("Stop immediately", False),
            ("Load new file", False),
            ("Stop after run", False),
            ("Cancel stop after run", False),
            ("Pause script", False),
            ("Resume script", False),
            ("Debug script", False),
            ("Step over", False)
        ])

        self.seq.odb_set(f"{self.odb_base}/Command", default_commands, update_structure_only=False)
        self.seq.odb_watch(f"{self.odb_base}/Command", self.command_changed, pass_changed_value_only=False)
        self.command_changed(None, None, self.seq.odb_get(f"{self.odb_base}/Command"))

        if not self.seq.odb_exists(f"{self.odb_base}/Param"):
            self.seq.odb_set(f"{self.odb_base}/Param", {"Value": {}, "Defaults": {}, "Comment": {}, "Options": {}})

        default_state = collections.OrderedDict([
            ("New file", False),
            ("Path", ""),
            ("Filename", ""),
            ("SFilename", ""),
            ("Next Filename", [""] * 10),
            ("Error", ""),
            ("Error line", 0),
            ("SError line", 0),
            ("Message", ""),
            ("Message Wait", False),
            ("Running", False),
            ("Finished", False),
            ("Paused", False),
            ("Debug", False),
            ("Current line number", 0),
            ("SCurrent line number", 0),
            ("Stop after run", False),
            ("Transition request", False),
            ("Loop counter", [0] * self.max_loop_odb),
            ("Loop n", [0] * self.max_loop_odb),
            ("Wait value", ctypes.c_float(0)),
            ("Wait limit", ctypes.c_float(0)),
            ("Start time", ctypes.c_uint32(0)),
            ("Wait type", ""),
            ("Wait ODB", ""),
            ("Last msg", "")
        ])
        self.seq.odb_set(f"{self.odb_base}/State", default_state, update_structure_only=True)

        self.seq.register_transition_callback(midas.TR_STOP, 999, self.end_of_run_handler)
        self.seq.register_disconnect_callback(self.disconnect_handler)

        logger.debug("Finished initialising SequencerRunner")

    def reset_state(self, including_params, including_error=True):
        """
        Reset both our internal state and the state in the ODB.

        Args:
            * include_params (bool) - also elete params and variables. They'll be 
                recreated when we run the user's `define_params()` function.
        """
        self.odb_update_period = datetime.timedelta(seconds=1)
        self.odb_update_last = datetime.datetime.now()
        self.latest_line_state = None
        self.latest_wait_info = None
        self.loop_states = []
        self.stop_after_run = False
        self.set_state_odb("Finished", False)
        self.set_state_odb("Running", False)
        self.set_state_odb("Paused", False)
        self.set_state_odb("Stop after run", False)
        self.set_state_odb("Message", "")
        self.set_state_odb("Message wait", False)
        self.set_state_odb("Transition request", False)
        self.set_state_odb("Wait type", "")
        self.set_state_odb("Wait value", 0)
        self.set_state_odb("Wait limit", 0)
        self.set_state_odb("Start time", 0)

        if including_error:
            self.report_error("", 0)

        if including_params:
            self.seq.odb_set(f"{self.odb_base}/Param/Value", {})
            self.seq.odb_set(f"{self.odb_base}/Param/Defaults", {})
            self.seq.odb_set(f"{self.odb_base}/Param/Comment", {})
            self.seq.odb_set(f"{self.odb_base}/Param/Options", {})
        
        self.seq.odb_set(f"{self.odb_base}/Variables", {})

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
        if hasattr(self, "seq"):
            self.seq.disconnect()

    def command_changed(self, client, path, odb_commands):
        """
        Callback function for when "/PySequencer/Command" is changed
        in the ODB. This function is very light/quick and just sets flags
        that will be acted on by the main `run()` and `run_sequence()` functions.

        Args:
            * odb_commands (dict) - current state of "/PySequencer/Command" in ODB
        """
        if odb_commands["Start script"]:
            self.start_script = True
            comms.debug = False
            self.set_command_odb("Start script", False)
            self.set_state_odb("Debug", False)

        if odb_commands["Stop immediately"]:
            comms.stop = True
            self.set_command_odb("Stop immediately", False)

        if odb_commands["Load new file"]:
            self.load_new_file = self.get_state_odb("Filename")
            # Reset ODB flag only once file has been loaded

        if odb_commands["Stop after run"]:
            self.stop_after_run = True
            self.set_command_odb("Stop after run", False)
            self.set_state_odb("Stop after run", True)

        if odb_commands["Cancel stop after run"]:
            self.stop_after_run = False
            self.set_command_odb("Cancel stop after run", False)
            self.set_state_odb("Stop after run", False)

        if odb_commands["Pause script"]:
            comms.pause = True
            self.set_command_odb("Pause script", False)
            self.set_state_odb("Paused", True)

        if odb_commands["Resume script"]:
            comms.pause = False
            comms.debug = False
            self.set_command_odb("Resume script", False)
            self.set_state_odb("Debug", False)
            self.set_state_odb("Paused", False)

        if odb_commands["Debug script"]:
            if self.exec_thread is None:
                # Start script only if not already running
                self.start_script = True

            comms.debug = True
            self.set_command_odb("Debug script", False)
            self.set_state_odb("Debug", True)
            self.set_state_odb("Paused", True)

        if odb_commands["Step over"]:
            comms.step_over = True
            self.set_command_odb("Step over", False)

    def disconnect_handler(self, client):
        """
        Try to stop sequence when user kills the main program.
        """
        comms.stop = True

        if self.exec_thread is not None:
            self.exec_thread.join()

    def end_of_run_handler(self, client, run_number):
        """
        Set a flag if the user asked us to stop the sequence when a run finished,
        and the run has just finished.
        """
        if self.stop_after_run:
            comms.stop = True

    def update_state_in_odb(self, force=False):
        """
        Called frequently to processes any messages in the state queue that
        were published by the other thread. Periodically we will make a summary 
        and writes that to the ODB.

        Also handles "Message wait" logic, which is in "State" rather than the
        more natural "Command" to match the MSL-based sequencer.
        """
        in_trans = bool(self.get_state_odb("Transition request"))
        new_in_trans = None

        # Handle webpage clearing the message
        if self.get_state_odb("Message wait"):
            if self.get_state_odb("Message") == "":
                comms.message_acknowledged = True
                self.set_state_odb("Message wait", False)

        # Handle any messages from execution thread
        while not comms.state_queue.empty():
            msg = comms.state_queue.get()

            if isinstance(msg, StateInfo):
                line_state = msg

                if line_state.filename == "<string>":
                    logger.debug(f"File {line_state.filename}, line {line_state.line_num}, func {line_state.func_name}, variables {line_state.variables}")
                    self.latest_line_state = line_state
                
                if line_state.filename == "client.py" and line_state.func_name in ["start_run", "stop_run", "pause_run", "resume_run"]:
                    # User called one of the transition functions from `midas.client.MidasClient()`.
                    # (Note we assume that the user doesn't have their own functions called "start_run" etc!)
                    logger.debug(f"See transition {line_state.func_name} starting")
                    new_in_trans = True
                elif (in_trans or new_in_trans) and line_state.func_name not in ["_run_transition"]:
                    # We've run another function, so the transition must be complete!
                    logger.debug(f"Transition complete {in_trans} {new_in_trans}")
                    new_in_trans = False
            elif isinstance(msg, LoopInfo):
                use_idx = None

                for idx, pb in enumerate(self.loop_states):
                    if pb.id == msg.id:
                        use_idx = idx
                        break

                if use_idx is None:
                    use_idx = len(self.loop_states)
                    self.loop_states.append(msg)

                if msg.what in ["early_exit", "complete"]:
                    self.loop_states = self.loop_states[:use_idx]
                else:
                    self.loop_states[use_idx] = msg
            elif isinstance(msg, WaitInfo):
                self.latest_wait_info = msg
            elif isinstance(msg, ScriptInfo):
                self.seq.odb_set(f"{self.odb_base}/Script/Lines", msg.lines)
            elif isinstance(msg, MessageInfo):
                self.set_state_odb("Message", msg.text)
                self.set_state_odb("Message wait", msg.wait)
            elif isinstance(msg, ResetParamsInfo):
                self.old_params = self.seq.odb_get(f"{self.odb_base}/Param")
                self.params = {"Value": {}, "Defaults": {}, "Comment": {}, "Options": {}}
            elif isinstance(msg, ParamInfo):
                if msg.name in self.old_params["Value"]:
                    self.params["Value"][msg.name] = self.old_params["Value"][msg.name]
                else:
                    self.params["Value"][msg.name] = msg.default_value

                self.params["Defaults"][msg.name] = msg.default_value
                self.params["Comment"][msg.name] = msg.comment

                if len(msg.options):
                    self.params["Options"][msg.name] = msg.options
            elif isinstance(msg, FinishedParamsInfo):
                self.seq.odb_set(f"{self.odb_base}/Param", self.params)
            elif isinstance(msg, ScriptLoadedInfo):
                self.set_command_odb("Load new file", False)
            elif isinstance(msg, ExceptionInfo):
                if isinstance(msg.e, StopSequencerException):
                    logger.info("Sequence was stopped by user")
                    self.set_state_odb("Stop after run", False)
                else:
                    logger.info(f"Sequence raised an exception during '{msg.when}' stage:")
                    logger.info("---------------------------------------------")
                    logger.info("".join(traceback.format_exception(msg.e, value=msg.e, tb=msg.e.__traceback__)))
                    logger.info("---------------------------------------------")

                    msg_str = f"{type(msg.e).__name__} during '{msg.when}' stage: {str(msg.e)}"
                    line = self.latest_line_state.line_num if msg.when == "sequence" else 0
                    self.report_error(msg_str, line)

        # Now actually update the ODB
        now = datetime.datetime.now()

        if new_in_trans is not None and in_trans != new_in_trans:
            # Update this one in ODB immediately, as state is trickier to track
            self.set_state_odb("Transition request", new_in_trans)

        if force or now - self.odb_update_last > self.odb_update_period:
            # Update everything else in ODB only every 1s or so
            self.odb_update_last = now

            if self.latest_line_state is None:
                self.set_state_odb("Current line number", 0)
                self.set_state_odb("SCurrent line number", 0)
                self.seq.odb_set(f"{self.odb_base}/Variables", {})
            else:
                self.set_state_odb("Current line number", self.latest_line_state.line_num)
                self.set_state_odb("SCurrent line number", self.latest_line_state.line_num)

                latest_vars = {k: self.get_odb_repr(v) for k,v in self.latest_line_state.variables.items()}
                latest_vars = {k: v for k,v in latest_vars.items() if v is not None}            
                klist = list(latest_vars.keys())
                vars_sorted = collections.OrderedDict([(k[:31], latest_vars[k]) for k in sorted(klist)])
                
                param_names = self.seq.odb_get(f"{self.odb_base}/Param/Defaults").keys()
                params_dict = {k: self.seq.get_param(k) for k in param_names}
                params_sorted = collections.OrderedDict([(f"Param: {k}"[:31], params_dict[k]) for k in sorted(param_names)])
                
                all_vars = params_sorted
                
                for k, v in vars_sorted.items():
                    all_vars[k] = v
                
                self.seq.odb_set(f"{self.odb_base}/Variables", all_vars)
                
            if self.latest_wait_info is None:
                self.set_state_odb("Wait type", "")
                self.set_state_odb("Wait ODB", "")
                self.set_state_odb("Wait value", 0)
                self.set_state_odb("Wait limit", 0)
                self.set_state_odb("Start time", 0)
            else:
                curr = self.latest_wait_info.curr_value
                targ = self.latest_wait_info.target_value

                if self.latest_wait_info.wait_type in ["Seconds", "ClientsRunning", "ODBValueStability"]:
                    # msequencer reports in ms rather than secs. Match that behaviour.
                    curr = round(curr * 1000, 0)
                    targ = round(targ * 1000, 0)

                self.set_state_odb("Wait type", self.latest_wait_info.wait_type)
                self.set_state_odb("Wait ODB", self.latest_wait_info.odb_path)
                self.set_state_odb("Wait value", curr if isinstance(curr, numbers.Number) else -1)
                self.set_state_odb("Wait limit", targ if isinstance(targ, numbers.Number) else -1)
                self.set_state_odb("Start time", self.latest_wait_info.started_ms)

            loop_counter = [0] * self.max_loop_odb
            loop_n = [0] * self.max_loop_odb

            for idx, loop_state in enumerate(self.loop_states):
                if idx < self.max_loop_odb:
                    loop_counter[idx] = loop_state.counter
                    loop_n[idx] = loop_state.n
            
            self.set_state_odb("Loop counter", loop_counter)
            self.set_state_odb("Loop n", loop_n)

            # Just copy the logic from msequencer here...
            recent_messages = self.seq.get_recent_messages(1)
            if len(recent_messages):
                self.set_state_odb("Last msg", recent_messages[0][11:19])

    def get_odb_repr(self, v):
        """
        Convert local/global variable into a string that can be shown in the ODB. 
        Only do it for certain types, as the user gets no benefit from seeing 
        '<class xyz>' etc.
        """
        # We store variables in ODB as strings
        report_types = [int, float, str, bool]

        if have_numpy:
            report_types.extend([np.number, np.bool])

        report_types = tuple(report_types)
        check_val = v
        retval = None
        max_display_len = 150
        is_np_arr = False

        if isinstance(v, list) and len(v) > 0:
            check_val = v[0]

        if have_numpy and isinstance(v, np.ndarray) and len(v) > 0:
            check_val = v[0]
            is_np_arr = True

        if isinstance(check_val, report_types):
            if is_np_arr and v.ndim == 1:
                try:
                    # Try to print 1D numpy arrays like a normal 1D python array (with commas)
                    retval = np.array2string(v, separator=', ').replace('\n', '')
                except:
                    retval = str(v)
            else:
                retval = str(v)

        if max_display_len > midas.MAX_STRING_LENGTH:
            max_display_len = midas.MAX_STRING_LENGTH

        if retval is not None and len(retval) >= max_display_len:
            # Too long to be stringified.
            if isinstance(v, list):
                retval = f"List of len {len(v)}: {v}"

            retval = retval[:max_display_len - 4] + "..."
        
        return retval
    
    def set_state_odb(self, k, v):
        """
        Wrapper to set an ODB key in /PySequencer/State.

        Args:
            * k (str) - ODB key
            * v (anything) - value to set
        """
        self.seq.odb_set(f"{self.odb_base}/State/{k}", v)

    def get_state_odb(self, k):
        """
        Get an ODB key from /PySequencer/State

        Args:
            * k (str)

        Returns:
            anything
        """
        return self.seq.odb_get(f"{self.odb_base}/State/{k}")

    def set_command_odb(self, k, v):
        """
        Wrapper to set an ODB key in /PySequencer/Command.

        Args:
            * k (str) - ODB key
            * v (anything) - value to set
        """
        self.seq.odb_set(f"{self.odb_base}/Command/{k}", v)

    def run_sequence(self, only_load=False, load_from_odb=False):
        """
        Spawn a new thread to load and/or run a sequence script.

        Handles spawning the thread, telling it to load/run the sequence, periodically
        updating the state in the ODB, and more.

        This function will only return once the spawned thread has finished (i.e. once
        the script has finished running, either because it reached the end of the script,
        because the user asked for the script to stop, or because the script raised an
        exception and terminated early).

        Args:
            * only_load (bool) - only load the script, don't execute it
            * load_from_odb (bool) - use the script that was most recently loaded
                and is currently stored in the ODB
        """
        comms.reset()

        threading.settrace(trace_calls)

        # Spawn the thread that will actually load/run the script.
        self.exec_thread = ScriptExecutionThread()

        # Figure out what script we'll run
        if load_from_odb:
            this_file = None
            this_path = None
            code_lines = self.seq.odb_get(f"{self.odb_base}/Script/Lines")
            self.reset_state(False)
        else:
            this_file = self.get_script_filename_from_odb()
            if not this_file:
                logger.warning("Aborting `run_sequence()`: filename is empty or invalid")
                return
            this_path = self.get_full_script_file_path(this_file)
            code_lines = []
            self.reset_state(False)

        if comms.debug:
            # We just reset these flags to False in "reset_state" - fix them
            self.set_state_odb("Paused", True)
            self.set_state_odb("Debug", True)

        self.exec_thread.set_args(self.seq, this_path, only_load, self.keep_modules, code_lines)
        self.set_state_odb("SFilename", this_path)
        
        # Start the spawned thread, which will load/run the script
        self.exec_thread.start()

        if not only_load:
            self.set_state_odb("Running", True)

        self.update_state_in_odb(True)

        # Wait until the spawned thread has finished its work
        while True:
            if self.exec_thread.is_alive():
                # `communicate()` will call command_changed() callback whenever
                # user sends a command via ODB.
                self.seq.communicate(100)

                self.update_state_in_odb()
            else:
                break

        # Tidy up
        self.update_state_in_odb(True)
        self.exec_thread.join()
        self.exec_thread = None

        self.reset_state(False, False)
        self.set_state_odb("Finished", True)

        self.update_state_in_odb(True)

    def get_script_filename_from_odb(self):
        """
        Get the path to the script that the user wants us to run.

        The user only gets to specify a relative path. This full path is
        <experiment_dir>/userfiles/sequencer/<user's path>. To get the full
        path, call `get_full_script_file_path()`.
        
        Returns:
            * str
        """
        path = self.get_state_odb("Path")
        file = self.get_state_odb("Filename")

        # return cleanly if Filename is empty
        if not file:
            return ""
        
        if ".." in path or ".." in file:
            self.report_error("Filenames/paths with '..' are not permitted", 0)
            return ""
        
        ext = os.path.splitext(file)[1]

        if ext != ".py":
            self.report_error(f"Filename must end with '.py', not '{ext}'")
            return ""
        
        return os.path.join(path, file)

    def get_full_script_file_path(self, file):
        """
        Get the full path to a sequencer script file.

        Args:
            * file (str) - path relative to <experiment_dir>/userfiles/sequencer

        Returns:
            str - the full file path
        """
        return os.path.join(self.seq.get_experiment_dir(), "userfiles", "sequencer", file)
    
    def report_error(self, msg, line):
        """
        Report an error to /PySequencer/State/Error.

        Args:
            * msg (str)
            * line (int) 
        """
        self.set_state_odb("Error", msg)
        self.set_state_odb("Error line", line)
        self.set_state_odb("SError line", line)

    def run(self):
        """
        Main function that waits for the user to tell us to do some work.

        When a script is executing we spend our time in `run_sequence()`.
        When a script is not executing, we spend our time in this function.
        """
        # Try to load the previous file if possible
        if self.get_script_filename_from_odb() != "":
            self.run_sequence(only_load=True)

        # Now loop forever, waiting for instructions from user
        while True:
            self.seq.communicate(100)

            if self.load_new_file is not None:
                self.run_sequence(only_load=True)
                self.load_new_file = None

            if self.start_script:
                # Run the first script
                self.start_script = False
                self.run_sequence(load_from_odb=True)

                while True:
                    # Run any scripts in "Next filename" list
                    next_filename = self.get_state_odb("Next filename")[0]

                    if next_filename == "":
                        break

                    logger.info("Moving on to next filename in list...")
                    filename_list = self.get_state_odb("Next filename")
                    self.set_state_odb("Filename", filename_list[0])
                    new_filename_list = filename_list[1:]
                    new_filename_list.append("")
                    self.set_state_odb("Next filename", new_filename_list)
                    self.run_sequence()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", type=str, metavar="name", default="", help="Name of additional sequencer. E.g. if 'Test', ODB location will be /PySequencerTest")
    parser.add_argument("-h", type=str, metavar="host_name")
    parser.add_argument("-e", type=str, metavar="expt_name")
    parser.add_argument("-D", action="store_true", help="Become a daemon")
    parser.add_argument("-O", action="store_true", help="Become a daemon but retain stdout")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-to-midas", action="store_true", help="Write to midas message log as well as stdout/stderr (only for logging.info() etc, not regular print() statements)")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    args = parser.parse_args()

    handler = midas.MidasLogHandler(facility="pysequencer") if args.log_to_midas else logging.StreamHandler()

    if args.verbose:
        formatter = logging.Formatter("%(asctime)s (%(filename)s:%(lineno)d:%(funcName)s) - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    with SequenceRunner(args) as runner:
        runner.run()
