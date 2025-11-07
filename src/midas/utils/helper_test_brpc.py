"""
This file is used in conjunction with the tests/test_brpc.py script.
See that file for more details.
"""

import midas.client
import sys

cmd = "make_blob"
file_path_ev = __file__ + ".event"
file_path_arb = __file__ + ".arbitrary"

if __name__ == "__main__":
    print("Creating this client")
    sys.stdout.flush()

    client = midas.client.MidasClient("pytest2")

    print("Connecting to other client")
    sys.stdout.flush()
    
    conn = client.connect_to_other_client("pytest")

    print("Calling as_event")
    sys.stdout.flush()

    ret_data = client.brpc_client_call(conn, cmd, "as_event", as_event=True)

    print("Writing as_event")
    sys.stdout.flush()

    with open(file_path_ev, "wb") as f:
        f.write(ret_data.pack())

    print("Calling arbitrary")
    sys.stdout.flush()

    ret_data = client.brpc_client_call(conn, cmd, "arbitrary", as_event=False)

    print("Writing arbitrary")
    sys.stdout.flush()

    with open(file_path_arb, "wb") as f:
        f.write(ret_data)

    print("Disconnecting")
    sys.stdout.flush()

    client.disconnect_from_other_client(conn)
    client.disconnect()
