#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2212/etc/bashrc
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
casename=$(basename $PWD)
touch case.foam
pvpython renderResults.py $casename
convert  $casename.png -transparent white -trim -resize 90% $casename.png
#curl -s --location --request POST "https://api.imgbb.com/1/upload?expiration=600&key=${IMGBB_API_KEY}"\
#    --form "image=@./$casename.png" | jq .data.url
curl -s --location --request POST "https://api.imgbb.com/1/upload?expiration=600&key=${IMGBB_API_KEY}"\
   --form "image=@./$casename.png" 
