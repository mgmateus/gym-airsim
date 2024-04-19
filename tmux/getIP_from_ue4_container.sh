#!/bin/bash

export UE4_IP="$(ping ue4 -c1 | head -1 | grep -Eo '[0-9.]{4,}')"