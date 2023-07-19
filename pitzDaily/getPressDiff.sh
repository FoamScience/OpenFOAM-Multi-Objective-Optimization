#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2012/etc/bashrc
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
time=$(foamListTimes)
if [ $time == "" ] || [ $time == "10000" ]; then
    echo "10000"
else
    pvpython pressureDiff.py $PWD $time
fi
