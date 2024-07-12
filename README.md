# Multi Objective Optimization on OpenFOAM cases

<a href="https://zenodo.org/record/7997394"><img src="https://zenodo.org/badge/611991004.svg"></a>

> If you're using this piece of software, please care enough to [cite it](https://zenodo.org/record/7997394) in your publications

Relying on [ax-platform](https://ax.dev) to experiment around 0-code parameter variation and multi-objective optimization
of OpenFOAM cases.

## Objectives and features
- Parameter values are fetched through a YAML/JSON configuration file. Absolutely no code should be needed, add parameters
  to the YAML file and they should be picked up automatically
- The no-code thing is taken to the extreme, through a YAML config file, you can (need-to):
  - Specify the template case
  - Specify how the case is ran
  - Specify how/where parameters are substituted
  - Specify how your metrics are computed

## How do I try this out?

Some [examples](examples), which range from simple and moderate levels of complexity, are provided
as reference.

Strictly speaking, you don't need an OpenFOAM installation unless you are running a CFD case. You
can always use your own code to evaluate the trials;  but parameters must be passed through an
OpenFOAM-like dictionary (See [single-objective opt. example](examples/local/single-objective)
for inspiration.

```bash
# Clone the repository
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization foamBO
cd foamBO
# Install dependencies
pip3 install -r requirements.txt
```
### Apptainer containers

```bash
git clone https://github.com/FoamScience/openfoam-apptainer-packaging /tmp/of_tainers
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization foamBO
cd foamBO
ansible-playbook /tmp/of_tainers/build.yaml --extra-vars="original_dir=$PWD" --extra-vars="@build/config.yaml"
# Get an apptainer container at containers/projects/foambo-<version>.sif
```

## Contribution is welcome!

By either filing issues or opening pull requests, you can contribute to the development
of this project, which I would appreciate.
