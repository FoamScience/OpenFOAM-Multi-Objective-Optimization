#/usr/bin/env python

"""
    A service-like script to start DB and stops, depending on messages received from
    other processes
"""

from smartsim import Experiment
from omegaconf import DictConfig
import zmq
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def db_main(cfg : DictConfig) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    
    exp = Experiment(name="dummyExp", launcher="local")
    db = exp.create_database(interface=cfg.interface, port=int(cfg.port))
    exp.start(db)
    print(f"dummyExp Database started ({cfg.interface}, {cfg.port})")
    print(f"Listening on tcp://*:5555 for requests to shutdown DB")
    
    message = socket.recv()
    print(f"Received request: {message} db")
    
    if message == b"stop":
        exp.stop(db)

if __name__ == "__main__":
    db_main()
