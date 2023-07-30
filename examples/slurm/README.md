# Shape optimization of a PitzDaily case

This example originates from a [ML/OpenFOAM Hackathon](https://github.com/OFDataCommittee/OFMLHackathon)
challenge. Head to the
[OFMLHackathon folder](https://github.com/FoamScience/OFMLHackathon/tree/main/2023-07/bayesian-optimization)
for a more detailed overview.

## Prerequisites

- You need the Python dependencies: `pip install -r requirements.txt`
- You need an OpenFOAM version. Set `FOAMBO_OPENFOAM` to its installation folder.
  `${FOAMBO_OPENFOAM}/etc/bashrc` will be sourced in cases' `Allrun`.
- A SLURM cluster (See [multi-objective opt. tutorial](../local/multi-objective/README)
  for an example cluster)
- Mesh parametrization is done through a `OpenSCAD -> cfMesh -> ParaView` workflow.

## PitzDaily with varying lowerWall geometry

The [config.yaml](shape-optimization/config.yaml) file shows how to setup the
[pitzDaily](shape-optimization/pitzDaily) for shape optimization using SAASBO. This simply
parameterizes the lower wall of the geometry as a cubic Bezier curve with 8 control
points, whose coordinates are used as optimization parameters.

## Dependent parameters

The [config-dependent.yaml](./shape-optimization/config-dependent.yaml) and the corresponding
case introduce a parameter for the number of control points, which then the coordinates depend
on.

You can run it with:
```bash
foamBO --config-name=config-dependent.yaml
```
