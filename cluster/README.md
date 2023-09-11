# Deploy with LXM3

Step 1: Compile requirements. 

```bash
hatch run compile:base
```
This generates a `requirements/base.txt` in the project root directory. 

Step 2: Build a Singularity image.
```bash
make build-singularity
``` 
This uses the requirements defined in Step 1. 

Step 3: Launch with LXM3. 
```
lxm3 launch launcher.py -- --config config/default.py --image d4rl_slim_benchmark-latest.sif --entrypoint d4rl_slim_benchmark.main 
```
