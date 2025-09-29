#!/usr/bin/env bash

# POST a SLURM job through the REST API
# Compatible with the docker cluster from https://github.com/FoamScience/hpc-makeshift-cluster
# and requires the head node to have an OpenFOAM apptainer container with pvpython support

casename=$(basename "$PWD")
cat <<EOF > "${casename}.json"
{
    "script": "#!/bin/bash\n${PWD}/Allrun /tmp/data/openfoam-paraview.sif",
    "job": {
        "environment": ["PATH=/bin/:/usr/bin/:/sbin/"],
        "name": "${casename}",
        "current_working_directory": "${PWD}",
        "tasks": 2
    }
}
EOF

export $(docker exec -it -u slurmuser slurm-head scontrol token | tr -d '\n\r')
export SLURM_REQ_URL=http://localhost:6820/slurm/v0.0.43

docker exec -u slurmuser slurm-head sudo chown -R slurmuser: "$PWD"
curl -s -X POST "$SLURM_REQ_URL/job/submit" \
    -H "X-SLURM-USER-NAME: slurmuser" \
    -H "X-SLURM-USER-TOKEN: $SLURM_JWT" \
    -H "Content-Type: application/json" \
    --data-binary "@${casename}.json"
