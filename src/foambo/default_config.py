from .common import VERSION
from typing import Dict, Any
import yaml

class RangeTag:
    yaml_tag = '!range'
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step
def range_representer(dumper, data: RangeTag):
    seq = [data.start, data.stop] if data.step == 1 else [data.start, data.stop, data.step]
    return dumper.represent_sequence('!range', seq, flow_style=True)
yaml.add_representer(RangeTag, range_representer)
def yaml_range(loader, node):
    values = loader.construct_sequence(node)
    return list(range(*values)) if len(values) > 1 else []
yaml.add_constructor("!range", yaml_range)
yaml.SafeLoader.add_constructor("!range", yaml_range)

def get_default_config() -> Dict[str, Any]:
    """
    Return a DictConfig object with the default configuration.
    """
    default = {
        # The foamBO version to run this configuration with
        "version": str(VERSION),
        # Ax Experiment setup
        "experiment": {
            # Experiment name, used as a prefix for trial case names
            "name": "Sample",
            # Experiment description
            "description": "Sample experiment description",
            # Parameter list, each parameter is defined with either a range or choice-values
            "parameters": [
                # if "name" only is supplied, required/optional entries will be listed
                {
                    "name": "x",
                    "bounds": [ 0.0, 1.0 ],
                    "step_size": None, # optional
                    "scaling": None,   # optional, "log"
                    "parameter_type": "float"
                },
                {
                    "name": "y",
                    "values": [ "zero", "one" ],
                    "is_ordered": None, # optional
                    "dependent_parameters": None, # optional
                    "parameter_type": "str"
                },
            ],
            # Linear constraints between numerical parameters
            "parameter_constraints": [] # as list of strings. eg "x >= y"
        },
        # Trial generation strategy
        "trial_generation": {
            "method": "custom", # or fast, random_search
            # generation nodes must be provided when custom is selected
            "generation_nodes": [
                {
                    # Node for a single trial at search space center
                    "next_node_name": "init",
                },
                {
                    # Node for a few randomly-sampled initialization trials
                    # Allowing for finer control
                    "node_name": "init",
                    # Eg. This will always generate the same random trials -> reproducible
                    "model_specs": [
                        {
                            "model_enum": "SOBOL",
                            "model_kwargs": {
                                "seed": 12345
                            }
                        }
                    ],
                    # Eg. Move over to next generator node after 7 trials in experiment
                    "transition_criteria": [
                        {
                            "type": "max_trials",
                            "threshold": 7,
                            "transition_to": "BOM",
                            "use_all_trials_in_exp": True
                        }
                    ]
                },
                {
                    # Node for a few randomly-sampled initialization trials
                    "node_name": "BOM",
                    "model_specs": [
                        {
                            "model_enum": "BOTORCH_MODULAR",
                        }
                    ],
                    # last node can have empty transition criteria
                    "transition_criteria": []
                },
            ]
        },
        # Include already-ran trial data from a CSV file
        # Format: x,y,metric,case_path
        # metric can be a scalar, or a (mean, sem) tuple
        # case_path column can be empty, the trial cases don't have to be on disc
        # rows that have the same parameter sets will be treated as progressions of same trial
        "existing_trials": {
            "file_path": ""
        },
        # This is an optional entry to set a "baseline" trial for BO analysis
        "baseline": {
            "parameters": {
                "x": 0.8,
                "y": "zero"
            } # if no baseline is to be set; set this to null
        },
        # BO settings
        "optimization": {
            # Metrics definition
            "metrics": [
                {
                    "name": "metric",
                    "progress": [ "echo", "$STEP" ], # optional for early-stopping, can be null
                    "command": [ "echo", "0" ], # always a local command
                },
            ],
            # Minimize 'metric', "-m1, -m2" will do a MOO minimizing m1 and m2
            # metrics can be also be weighted: "-m1, -2*m2"
            "objective": "-metric",
            # Linear objective constraints, comma seperated, and can use "baseline"
            "outcome_constraints": ["metric >= 0.9*baseline"],
            # OpenFOAM case handling
            "case_runner": {
                # The template OpenFOAM case, needs to be a fully working one
                "template_case": "./case",
                # Case run mode
                "mode": "remote", # or local
                # How to run a case. Supports $CASE_PATH and $CASE_NAME replacements
                # CWD is always the trial's folder
                "runner": "./run_on_cluster.sh $CASE_NAME",
                # Should STDout of runner command be logged into a file inside the trial's folder?
                "log_runner": False,
                # How to check for case successful completion in case mode==remote
                # Supports $CASE_PATH and $CASE_NAME replacements
                # CWD is always the trial's folder
                "remote_status_query": "./state_on_cluster.sh $CASE_NAME",
                # How to kill remote job if wanting to early-stop the trial
                # Supports $CASE_PATH and $CASE_NAME replacements
                # CWD is always the trial's folder
                "remote_early_stop": "scancel --name $CASE_NAME",
                # Folder to store trial cases 
                "trial_destination": "./trials",
                # Folder to store client state and optimization reports
                "artifacts_folder": "./artifacts",
                # How to substitue parameter values into the case dictionaries
                # file substitution always precedes variable substitution
                "file_substitution": [
                    # will replace files depending on chosen y value for the trial:
                    # <trial>/constant/y.zero -> <trial>/constant/y
                    # <trial>/constant/y.one -> <trial>/constant/y
                    {
                        "parameter": "y",
                        "file_path": "/constant/y",
                    },
                ],
                "variable_substitution": [
                    {
                        # substitues x value in <trial>/0orig/field with chosen x parameter value
                        "file": "/0orig/field",
                        "parameter_scopes": {
                            "x": "someDict.x"
                        }
                    }
                ],
            }
        },
        # Optimization trial handling
        "orchestration_settings": {
            # Run at most 20 total experiment trials
            "max_trials": 20,
            # Run at most 3 trials at the same time
            "parallelism": 3,
            # How many seconds to wait to poll trials initially
            "initial_seconds_between_polls": 60,
            # Enlarge/shrink initial waiting time between polls gradually
            "seconds_between_polls_backoff_factor": 1.5,
            # Timeout in hours for the whole simulation
            "timeout_hours": 48,
            # Timeout in seconds for single trials
            "ttl_seconds_for_trials": 2400,
            # When to stop the optimization (only improvement-based strategy is supported)
            "global_stopping_strategy": {
                "min_trials": 10,
                "window_size": 5,
                "improvement_bar": 0.1
            },
            # When to early-stop a trial
            "early_stopping_strategy": {
                # Will look at all trials that have data for current trial's "progression-step"
                # if metric falls bellow 25% for these trials for the current step
                # the trial will be stopped
                "type": "percentile",
                "metric_names": [
                    "metric"
                ],
                "percentile_threshold": 25,
                # progression's steps are number of "poll" attempts
                "min_progression": 5,
                # Activate only after certain number of trials
                # The "range" constructor forwards values to python's range
                "trial_indices_to_ignore": RangeTag(0, 10, 1)
            }
        },
        # Experiment saving and loading
        "store": {
            "save_to": "json", # will save to artifacts/<experiment-name>_client_state.json, other options: sql
            "read_from": "json", # will read from same save location, other options: sql, nowhere
            "backend_options": {
                "url": None
            }
        }
    }
    return default
