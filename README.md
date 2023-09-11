# D4RL Slim Benchmark

<p align="center">
    <em>Code for benchmarking D4RL-slim</em>
</p>

[![build](https://github.com/dtch1997/d4rl-slim-benchmark/workflows/Build/badge.svg)](https://github.com/dtch1997/d4rl-slim-benchmark/actions)
[![codecov](https://codecov.io/gh/dtch1997/d4rl-slim-benchmark/branch/master/graph/badge.svg)](https://codecov.io/gh/dtch1997/d4rl-slim-benchmark)
[![PyPI version](https://badge.fury.io/py/d4rl-slim-benchmark.svg)](https://badge.fury.io/py/d4rl-slim-benchmark)

---

**Documentation**: <a href="https://dtch1997.github.io/d4rl-slim-benchmark/" target="_blank">https://dtch1997.github.io/d4rl-slim-benchmark/</a>

**Source Code**: <a href="https://github.com/dtch1997/d4rl-slim-benchmark" target="_blank">https://github.com/dtch1997/d4rl-slim-benchmark</a>

---

## Usage

### Setup environment

We use [Hatch](https://hatch.pypa.io/latest/install/) to manage the development environment and production build. Ensure it's installed on your system.

```bash
$ hatch run python -m template_demo.main --config template_demo/config/default.py 

seed: 0
track: false
wandb_entity: dtch1997
wandb_name: null
wandb_project: template_demo

```



## Development

### Run unit tests

You can run all the tests with:

```bash
hatch run test
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
hatch run lint
```

### Publish a new version

You can bump the version, create a commit and associated tag with one command:

```bash
hatch version patch
```

```bash
hatch version minor
```

```bash
hatch version major
```

Your default Git text editor will open so you can add information about the release.

When you push the tag on GitHub, the workflow will automatically publish it on PyPi and a GitHub release will be created as draft.

## Serve the documentation

You can serve the Mkdocs documentation with:

```bash
hatch run docs-serve
```

It'll automatically watch for changes in your code.

## License

This project is licensed under the terms of the MIT license.
