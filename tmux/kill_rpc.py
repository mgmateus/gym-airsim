#!/usr/bin/env python3

import msgpackrpc
import os

if __name__ == '__main__':
    host = os.environ['UE4_IP']
    port = 41451
    msgpackrpc.Client(msgpackrpc.Address(host=host, port=port)).close()
