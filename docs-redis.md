# Using SmartRedis DBs for metric evaluation 

First, go through [docs.md](docs.md).

We are using the dummy SLURM cluster to implement a metric evaluation function which
will use `AI.TENSOR`s from a Redis database to compute a metric's value.

For this we need a few components:
1. A Redis database server
2. An OpenFOAM function object which uses SmartRedis client to send fields to a Redis DB. The provided
   function object simply extracts `Ux` at cells next to the `lowerWall` patch and sends them together
   with `x` components of the cell centers to the Redis DB.
3. A Python function to compute current separation position in `pitzDaily` in `core.py`
  (already implemented for serial case runs; fetches `Ux` and `x` from Redis DB and looks for last sign change!)

For the Redis server, since we're on the Docker cluster, it's better if we add one more container to
host the database server. We would start the server service and wait until stopping is requested.

```bash
# Note how we specify the cluster's network.
# --rm makes the container get destroyed immediately when we leave it
# It also has access to var/axc like other cluster nodes
docker run -it --rm --name redis-db --net slurm-cluster_default -v $PWD:/axc ghcr.io/craylabs/smartsim-tutorials:v0.4.1 bash
# Get its IP Address in the cluster's network for later
docker inspect redis-db | jq '.[0].NetworkSettings.Networks."slurm-cluster_default".IPAddress'
# We assume it's 127.23.0.8 here
# Start the Redis server inside the container:
pip install omegaconf zmq hydra-core
cd /axc/multiOptOF
# Adapt the address if it's different for you
python redis-db.py +interface=eth0 +port=6532
```

For the OpenFOAM function object; you need to compile it on all nodes (Only on the head-node in this example).
```bash
cd /axc/multiOptOF
git clone https://github.com/CrayLabs/SmartRedis
# Execute all of the following commands on all nodes if smartSimFunctionObject needs to run on compute nodes too
openfoam2206
conda activate of-opt # This might need to be created
pip install smartsim[ml]
# We don't need PyTorch and TensorFlow backend for this simple example, so don't build them
yum install -y git-lfs
smart build --device cpu  --no_pt --no_tf
. bashrc.smartredis
cd SmartRedis
make lib
cd ..
wmake smartSimFunctionObject
# This is the db_server:db_port
export SSDB="172.23.0.8:6532"
```

You're all set; `./multiObjOpt.py` on the head node should start the sample optimization run.

You can connect to the database directly to monitor fields:
```bash
docker exec -it redis-db bash
/home/craylabs/.local/lib/python3.8/site-packages/smartsim/_core/bin/redis-cli -h 172.23.0.8 -p 6532
```

When done, you can send a signal to the DB server to gracefully stop with (this can be done from any
cluster node if you install dependencies there):
```bash
docker exec -it redis-db bash
cd /axc/multiOptOF
python stopdb.py
```

> Deploying something like this on the SLURM cluster is a matter of tweaking the function object so
> it either:
> - Combines info from all processors and send the result
> - Send individual processor results separately, but suffix the `procID`; The evaluation function
>   needs to take care of the rest.
