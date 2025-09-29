#!/usr/bin/env bash
# Get a metric value from the filling log file
# runs in both local and remote setups, both using a container
# Local Usage: ./metric_from_log.sh local vorticityTopRight
# Remote Usage: ./metric_from_log.sh remote vorticityTopRight

set -e
casename=$(basename "$PWD")
mode="$1"
casecwd=""
logfile=""
cmdprefix=""

case "$mode" in
    local)
        casecwd="$PWD"
        logfile="$casecwd/log.icoFoam"
        if [ ! -f "$logfile" ]; then echo nan; exit 1; fi
        ;;
    remote)
        casecwd="/tmp/data/trials/$(basename "$PWD")"
        logfile="$casecwd/log.icoFoam"
        docker exec slurm-head bash -c "if [ ! -f $logfile ]; then echo nan; exit 1; fi"
        cmdprefix="docker exec slurm-head "
        ;;
    *)
        echo "Unknown mode: $mode"
        echo "Supported modes are: local, remote"
        exit 1
        ;;
esac

case $2 in
    executionTime)
        awk_script='/^ExecutionTime = / { if ($3 > max) max = $3; found = 1 } END { if (found) print max; else print "nan" }'
        $cmdprefix bash -c "awk -v max=-100 '$awk_script' $logfile"
    ;;
    continuityErrors)
        awk_script='/^time step continuity errors :/ { if ($NF > max) max = $NF; found = 1 } END { if (found) print max*1.0e16; else print "nan" }' 
        $cmdprefix bash -c "awk -v max=-100 '$awk_script' $logfile"
    ;;
    wallShearStress)
        awk_script='/Sum of forces/{inF=1; next} /Sum of moments/{inF=0} inF && /Viscous/ { gsub("[()]",""); Fx=$2; Fy=$3; Fz=$4; shear=sqrt(Fx*Fx + Fy*Fy + Fz*Fz) } END {print shear}'
        $cmdprefix bash -c "awk '$awk_script' $logfile"
    ;;
    enstrophy)
        container_cmd="pvpython /tmp/data/scripts/pv_vorticity.py --enstrophy $casecwd"
        $cmdprefix bash -c "apptainer run /tmp/data/openfoam-paraview.sif '$container_cmd'" 2>/dev/null || echo "nan"
    ;;
    *)
        echo 'Unknown metric: `'"$2"'`'
        echo "Supported metrics: executionTime, continuityErrors, wallShearStress, enstrophy"
        exit 1
    ;;
esac
