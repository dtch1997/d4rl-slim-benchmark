# Deploy with LXM3

Step 1: Compile requirements.

The requirements for D4RL and d4rl-slim are different and incompatible, so we compile two separate sets of requirements. 

```bash
hatch run compile:d4rl
hatch run compile:d4rl-slim 
```
This generates a `requirements/{d4rl,d4rl-slim}.txt` in the project root directory.

Step 2: Build a Singularity image.

We have two different Dockerfiles because D4RL requires MuJoCo-Py, which requires some additional installation steps in the Dockerfile. 
```bash
make build-singularity IMAGE=d4rl DOCKERFILE=Dockerfile_d4rl
make build-singularity IMAGE=d4rl-slim DOCKERFILE=Dockerfile_d4rl_slim
```
This uses the requirements defined in Step 1.

Step 3: Launch with LXM3.

D4RL-slim experiments: 
```
lxm3 launch launcher.py -- --config config/offline/cql.py --entrypoint d4rl_slim_benchmark.train --image d4rl-slim-latest.sif --launch_on_cluster True --config.use_d4rl_slim True
```

D4RL experiments: 
```
lxm3 launch launcher.py -- --config config/offline/cql.py --entrypoint d4rl_slim_benchmark.train --image d4rl-latest.sif --launch_on_cluster True
```

