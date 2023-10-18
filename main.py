import zmq
import argparse

import AI_stream

if __name__ == "__main__":
    # Parse cmd arguments -> HoloLens IP Address
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="Indicate the HoloLens IP address")
    args = parser.parse_args()

    # Initialize TCP communications
    context = zmq.Context()
    client_socket = context.socket(zmq.REP)
    client_socket.bind("tcp://*:5555")

    AIServer = AI_stream.AIServer(args.host)

    # Start requests server
    while(True):
        print("Server ready for requests")  # Start, SetActiveClassList, GetObjectList or Stop
        request = client_socket.recv()  # Code blocks here until a request is received

        print(f"Received request: {request.decode('UTF-8')}")  # decode from bytes to str

        # Turn on sensor streaming
        #if request.decode('UTF-8') == "Start":
        if request.decode('UTF-8') == "Start":
            response = AIServer.ServerStart()

        # If header is ACL, set ActiveClassList
        elif request.decode('UTF-8').split(' ')[0] == "ACL":
            response = AIServer.ServerSetActiveClassList(request.decode('UTF-8').split(' ')[1:])

        # If header is OL, return the ObjectList with requested information
        elif request.decode('UTF-8').split(' ')[0] == "OL":
            response = AIServer.ServerGetObjectList(request.decode('UTF-8').split(' ')[1:])

        # Turn off sensor streaming
        elif request.decode('UTF-8') == "Stop":
            response = AIServer.ServerStop()

        else:
            response = "Unknown command"

        # Answer to the client
        print(response)
        client_socket.send_string(response)










