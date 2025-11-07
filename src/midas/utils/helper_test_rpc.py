"""
This file is used in conjunction with the tests/test_rpc.py script.
See that file for more details.
"""

import midas.client

cmd = "ABC"
args = "DEF GHI"
max_len = 100
file_path = __file__ + ".output"

def rpc_retstr(cmd, args, max_len):
    """
    String we'll be returning from the RPC function.
    """
    return "Hello %s %s %s" % (cmd, args, max_len)

def rpc_call_helper(cmd, args, max_len):
    """
    Call the RPC function registered by the "pytest" client.
    """
    client = midas.client.MidasClient("pytest2")
    conn = client.connect_to_other_client("pytest")
    retstr = client.jrpc_client_call(conn, cmd, args, max_len)
    client.disconnect_from_other_client(conn)
    client.disconnect()
    return retstr

if __name__ == "__main__":
    retstr = rpc_call_helper(cmd, args, max_len)
    with open(file_path, "w") as f:
        f.write(retstr)