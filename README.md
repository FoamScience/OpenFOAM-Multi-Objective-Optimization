# Multi Objective Optimization on OpenFOAM cases

<a href="https://zenodo.org/record/7997394"><img src="https://zenodo.org/badge/611991004.svg"></a>

> If you're using this piece of software, please care enough to [cite it](https://zenodo.org/record/7997394) in your publications

Relying on [ax-platform](https://ax.dev) to experiment around 0-code parameter variation and multi-objective optimization
of OpenFOAM cases.

## Objectives and features
- Parameter values are fetched through a YAML/JSON configuration file. Absolutely no code should be needed, add parameters
  to the YAML file and they should be picked up automatically
- The no-code thing is taken to the extreme, through a YAML config file, you can (need-to):
  - Specify the template case. This must be a "complete" (no placeholder), and runnable (as-is) OpenFOAM case.
  - Specify how the case is ran. Locally, or on SLURM, and specifying the commands to run.
  - Specify how/where parameters are substituted. `PyFoam` is used for this purpose for now.
  - Specify how your metrics are computed. These are just shell commands that output a single scalar
    value to `STDOUT`.

## How do I try this out?

Some [examples](examples), which range from simple and moderate levels of complexity, are provided
as reference.

Strictly speaking, you don't need an OpenFOAM installation unless you are running a CFD case. You
can always use your own code to evaluate the trials;  but parameters must be passed through an
OpenFOAM-like dictionary (See [single-objective opt. example](examples/local/single-objective)
for inspiration.

```bash
# Install the package
pip install foambo
# Clone the repo to get the examples
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization foamBO
cd foamBO/examples
```
### Apptainer containers

If you'd like to try this out in Apptainer containers instead, It's recommended to
(Must have `ansible` installed for this to work):

```bash
git clone https://github.com/FoamScience/openfoam-apptainer-packaging /tmp/of_tainers
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization foamBO
cd foamBO
ansible-playbook /tmp/of_tainers/build.yaml --extra-vars="original_dir=$PWD" --extra-vars="@build/config.yaml"
# Get an apptainer container at containers/projects/foambo-<version>.sif
```

Each example has some description in its README:
- [Single-Objective BO of a single-parameter function, ran locally](examples/local/single-objective/README.md)
- [Multi-Objective BO of a `pitzDaily` case, ran locally and on a SLURM Cluster](examples/local/multi-objective/README.md)
- [Multi-Objective BO of a `pitzDaily` case, ran on a SLURM cluster](examples/slurm/README.md)
  and providing "dependent parameters setup" variants.

## Contribution is welcome!

By either filing issues or opening pull requests, you can contribute to the development
of this project, which I would appreciate.
