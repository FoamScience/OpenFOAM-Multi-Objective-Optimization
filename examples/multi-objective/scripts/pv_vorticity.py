#!/usr/bin/env pvpython
# Computes average vorticity in top-right corner
# Usage: pvpython <this_script> [--corner-vorticity | --total-recirculation] <case_path>
# trace generated using paraview version 5.11.2
import os, sys, math, argparse
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
import paraview.servermanager as servermanager
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def main():
    parser = argparse.ArgumentParser(description="Compute vorticity-related metrics.")
    parser.add_argument("case_path", type=str, help="OpenFOAM case path")
    parser.add_argument("--enstrophy", action='store_true', help="An integral measure of small-scale rotational energy")
    args = parser.parse_args()

    # create a new 'OpenFOAMReader'
    casefoam = OpenFOAMReader(registrationName='case.foam', FileName=f'{args.case_path}/case.foam')
    
    animationScene1 = GetAnimationScene()
    animationScene1.UpdateAnimationUsingDataTimeSteps()
    UpdatePipeline(time=1.0, proxy=casefoam)
    
    if args.enstrophy:
        calculator1 = Calculator(registrationName='Calculator1', Input=casefoam)
        calculator1.AttributeType = 'Cell Data'
        calculator1.ResultArrayName = 'magVorticity'
        calculator1.Function = 'mag(vorticity)^2'
        UpdatePipeline(time=1.0, proxy=calculator1)
        integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=calculator1)
        integrateVariables1.DivideCellDataByVolume = 1
        UpdatePipeline(time=1.0, proxy=integrateVariables1)
        data = servermanager.Fetch(integrateVariables1)
        print(data.GetCellData().GetArray("magVorticity").GetValue(0)/1000.0)

if __name__ == '__main__':
    main()
