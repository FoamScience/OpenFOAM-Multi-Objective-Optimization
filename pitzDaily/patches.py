#!/opt/paraviewopenfoam510/bin/pvpython
import os
import sys
#print(sys.version)

import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
import paraview.servermanager as servermanager
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'STL Reader'
mainstl = STLReader(registrationName='pitzDaily2D.stl', FileNames=[sys.argv[1] + '/pitzDaily2D.stl'])

# create a new 'Generate Surface Normals'
generateSurfaceNormals1 = GenerateSurfaceNormals(registrationName='GenerateSurfaceNormals1', Input=mainstl)

# create a new 'Connectivity'
connectivity1 = Connectivity(registrationName='Connectivity1', Input=generateSurfaceNormals1)
data = servermanager.Fetch(connectivity1)
rng = data.GetCellData().GetArray('RegionId').GetRange()
print(rng[0], rng[1])

i = 1
for j in range(int(rng[0]), int(rng[1]) + 1):
    threshold = Threshold(registrationName='Threshold_' + str(j), Input=connectivity1)
    threshold.Scalars = ['POINTS', 'RegionId']
    threshold.ThresholdRange = [j,j]
    data = servermanager.Fetch(threshold)
    patchNamei = "patch_" + str(i)
    if data.GetNumberOfCells() > 1:
        extractSurface = ExtractSurface(registrationName='ExtractSurface_' + str(j), Input=threshold)

        patchesToLookFor = {}
        functionToLook = {"upperWall": "coordsY > 1.5e-2", "inlet": "coordsX < -0.02", "outlet": "coordsX > 0.28"}

        
        for k in functionToLook.keys():
            calculator1 = Calculator(registrationName='Calculator' + str(k) + str(i), Input=extractSurface)
            resultName = 'Res' + str(k) + str(i)
            calculator1.ResultArrayName = resultName
            calculator1.Function = functionToLook[k]
            arr = servermanager.Fetch(calculator1)
            flds = arr.GetPointData()
            fld = flds.GetArray(resultName)
            isFound = True
            for ii in range(fld.GetSize()):
                if fld.GetValue(ii) < 1.0:
                    isFound = False
            if isFound:
                patchesToLookFor[patchNamei] = k
        
        #patchName = patchesToLookFor.get("patch_" + str(i), "patch_" + str(i))
        patchName = patchesToLookFor[patchNamei] if patchNamei in patchesToLookFor.keys() else patchNamei

        print("Writing " + patchName + ".stl ...")
        SaveData(sys.argv[1] + '/' + patchName + '.stl',
                 proxy=extractSurface,
                 #PointDataArrays=[],
                 #CellDataArrays=[],
                 FileType='Ascii')
        
        i += 1
