# Multi Objective Optimization on OpenFOAM cases

<a href="https://zenodo.org/record/7997394"><img src="https://zenodo.org/badge/611991004.svg"></a>

> If you're using this piece of software, please care enough to [cite it](https://zenodo.org/record/7997394) in your publications

Relying on [ax-platform](https://ax.dev) to experiment around 0-code parameter variation and multi-objective optimization
of OpenFOAM cases.

## Documentation

Please consult the [documentation page](/docs) for the specifics although you can see what configuration we support by browsing:
```bash
uvx foamBO --docs
```

## How do I try this out?

Some [examples](examples), which range from simple and moderate levels of complexity, are provided for reference.

Strictly speaking, you don't need an OpenFOAM installation unless you are running a CFD case. You can always use your own code to evaluate the trials;  but parameters must be passed through an OpenFOAM-like dictionary (See [single-objective opt. example](examples/single-objective) for inspiration.

```bash
# Install the package
pip install foambo
# Clone the repo to get the examples
git clone https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization foamBO
cd foamBO/examples/single-objective
foamBO --config SOM.yaml
```

## Contribution is welcome!

By either filling issues or opening pull requests, you can contribute to the development of this project, which I would appreciate.
