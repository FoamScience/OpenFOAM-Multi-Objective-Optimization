#!/usr/bin/env bash
source /fsx/OpenFOAM/OpenFOAM-v2212/etc/bashrc
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
time=$(foamListTimes)
if [[ $time == "" ]] || [[ $time == "10000" ]]; then
    echo "10000"
else
    #pvpython pressureDiff.py $PWD $time
    filename="postProcessing/pressureDifferencePatch/0/multiFieldValue.dat" 
    awk '
    END {
        value = $NF
        print "", value
    }
' "$filename"
fi
