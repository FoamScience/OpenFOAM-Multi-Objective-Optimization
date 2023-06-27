#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --time=0-00:10:00

cd "$1"
source /usr/lib/openfoam/openfoam2206/etc/bashrc
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------

set -e
touch case.foam
cp -rT 0orig 0
runApplication blockMesh
runApplication decomposePar

# Serial run
mpirun --allow-run-as-root --output-filename log $(getApplication) -parallel
