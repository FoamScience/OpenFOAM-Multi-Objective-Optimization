#!/usr/bin/env bash

# Get state of clutser job ran with run_on_cluster.sh script
# through SLURM's REST API

casename=$(basename "$PWD")
export $(docker exec -it -u slurmuser slurm-head scontrol token | tr -d '\n\r')
export SLURM_REQ_URL=http://localhost:6820/slurm/v0.0.43
curl -s -X GET "$SLURM_REQ_URL/jobs" \
    -H "X-SLURM-USER-NAME: slurmuser" \
    -H "X-SLURM-USER-TOKEN: $SLURM_JWT" \
    | jq '.jobs[] | select(.name == "'"${casename}"'") | .job_state[-1]'
