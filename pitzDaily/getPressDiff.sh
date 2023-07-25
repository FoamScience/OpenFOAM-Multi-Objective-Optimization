#!/usr/bin/env bash
source /fsx/OpenFOAM/OpenFOAM-v2212/etc/bashrc
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
time=$(foamListTimes)
penalty="100000"
# If there are no times written, penalize the pressure drop objective
if [[ $time == "" ]] ; then
    echo $penalty
else
    convergedP=$(awk '//{converged=$NF} END{print(converged)}' postProcessing/solverInfo/0/solverInfo.dat)
    case $convergedP in
        (true) pvpython pressureDiff.py $PWD $time;;
        (*) echo penalty;;
    esac
fi
