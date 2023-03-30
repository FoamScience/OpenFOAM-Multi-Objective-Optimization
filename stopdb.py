#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases using FullyBayesianMOO if possible

This script defines functions to perform multi-objective optimization on OpenFOAM
cases given a YAML/JSON config file (Supported through Hydra, default: config.yaml).

We use the Adaptive Experimentation Platform for optimization, PyFOAM for parameter substitution
and Hydra for 0-code configuration.

Output: CSV data for experiment trials

Things to improve:
- Optimization restart? Maybe from JSON file as a start.
- Dependent parameters.

Notes:
- You can also use a single objective
"""

import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.23.0.8:5555")

if __name__ == "__main__":
    socket.send(b"stop")
