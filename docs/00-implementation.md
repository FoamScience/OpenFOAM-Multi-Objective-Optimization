# Code design style

## Configuration validation and reporting of possible flags:

## Motivation and Effects

We take a banana-trick-like approach for user convenience when creating configuration files.
The goal is to make it easy for users and LLMs to reach a working configuration with trial-and-error.

To start, we get a missing top-level key error if it's not there yet. Let's take `orchestration_settings` as an example:

```shell
omegaconf.errors.ConfigKeyError: Missing key orchestration_settings
    full_key: orchestration_settings
    object_type=dict
```

If we then assume users don't know what parameters to set in the `orchestration_settings` sub-dict,
figuring it out is as easy as setting:

```yaml
orchestration_settings: null
```

```shell
ValueError: Got an empty configuration for OrchestratorOptions.__init__, but expecting fields
['max_trials', 'parallelism', 'global_stopping_strategy', 'tolerated_trial_failure_rate=0.5', 'initial_seconds_between_polls=0.1']
```

> Note that defaulted arguments are optional, but it's convenient to be able to see the active defaults.

This carries on recursively through the configuration tree. For example, the following configuration:
```yaml
# tolerated_trial_failure_rate and initial_seconds_between_polls are defaulted
orchestration_settings:
  max_trials: 30
  parallelism: 4
  global_stopping_strategy: null
```
will result in the following error:
```shell
TypeError: OrchestratorOptions.create_global_stopping_strategy() got unexpected arguments from global_stopping_strategy sub-dict.
Provided: empty configuration
Expected: ['min_trials', 'window_size', 'improvement_bar']
```

`src/foambo/common.py` defines some utilities to create objects straight from dictionaries and
ensure arguments are validated properly. Two usage patterns are common:


### Validate dictionary entries as function arguments

```python
# Will throw an exception if generation_config is missing required
# keys for it to unpack to client.configure_generation_strategy arguments
validate_args(client.configure_generation_strategy, generation_config)
client.configure_generation_strategy(**generation_config)
```

### Create objects with validation for nested fields creation

```python
@dataclass
class MyType():
    setting: int
    nested_field: AnotherType

    __nested_fields__ : {
        "nested_field": "create_nested_field"
    }

    @classmethod
    def create_nested_field(cls, arg1: int, arg2: float):
        return AnotherType(arg1=arg1, arg2=arg2)

obj = instantiate_with_nested_fields(MyType, cfg['my_type'])
```

This then can be helpful in creating documentation entities for each configuration entry.
