from paraview.simple import *
import paraview.servermanager as servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys
from pathlib import Path
import os
import argparse


#virtualEnvPath = sys.argv[sys.argv.index('--virtual-env') + 1]
virtualEnv = '/home/local/CSI/fb12pisy/paraview/bin/activate_this.py'
exec(open(virtualEnv).read(), {'__file__': virtualEnv})

try:
    from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
except:
    from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

# class in cpp, parses the argument list
parser = argparse.ArgumentParser()

#adding argument for the velocity, and it expect that argument
parser.add_argument('--We', action='store_true', help='get Weber number')
parser.add_argument('--Bo', action='store_true', help='get Bond number')
parser.add_argument('--L', action='store_true', help='get dimless length')
parser.add_argument('--i', action='store_true', help='bool, 1 if droplet passes, 0 if not')
parser.add_argument('--n', action='store_true', help='number of droplets')



# call to parse into argument
args = parser.parse_args()

# Get the density and surface tension
# Get current directory
current_dir = Path.cwd()

# Specify the path to the dictionary file
trspProp_path = "./constant/transportProperties"

# Specify the path to the dictionary file
g_path = "./constant/g"
dictionary_g = ParsedParameterFile(g_path)
g = dictionary_g['g'][0]

# Load the dictionary
dictionary_tp = ParsedParameterFile(trspProp_path)


# Specify the keyword you want to retrieve

rho = dict(dictionary_tp['phases'][1])['rho'][2]
#print(f"The value of rho is: {rho} kg/m3")

sigma = dictionary_tp['sigma'][2]
#print(f"The value of sigma is: {sigma} N/m")

# Get value of the initial droplet radius and defect width
# Specify the path to the dictionary file
BC_path = "./0.orig/theta0WaterAir"

# Load the dictionary
dictionary_bc = ParsedParameterFile(BC_path)
Lh   = dictionary_bc['l']
R0 = dictionary_bc['R']
#print(f"The value of l is: {Lh} m")
#print(f"The value of R0 is: {R0} m")

current_dir = Path.cwd()

# Change directory
#os.chdir(filename)

# Create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName='*.foam')
casefoam.CellArrays = ['CWater']
casefoam.CaseType = 'Decomposed Case'

previous_center_of_mass = [0.0, 0.0, 0.0]
previous_time = 0

# Load controlDict
controlDict_path = "./system/controlDict"
dictionary_CD = ParsedParameterFile(controlDict_path)
start_time = dictionary_CD['startTime']
end_time = dictionary_CD['endTime']
write_interval = dictionary_CD['writeInterval']

# start_time = 0
# end_time = 0.095
# write_interval = 0.005
available_time_steps = [t for t in np.arange(start_time, end_time + write_interval, write_interval)]

# Initialize an empty list to store velocities
velocities = []

# Loop over time steps
for time_step in available_time_steps:
    UpdatePipeline(time=time_step, proxy=casefoam)

    # Create a new 'Plot Over Line'
    plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=casefoam)
    plotOverLine1.Point1 = [-0.01, 0.0, 0.0]
    plotOverLine1.Point2 = [0.03, 0.0, 0.0]
    UpdatePipeline(time=time_step, proxy=plotOverLine1)
    SetActiveSource(casefoam)

    plotOverLineData = servermanager.Fetch(plotOverLine1)
    CWater_values = vtk_to_numpy(plotOverLineData.GetPointData().GetArray('CWater'))
    points = vtk_to_numpy(plotOverLineData.GetPoints().GetData())

    # Find the indices where the sign changes from positive to negative
    positive_to_negative_indices = np.where((CWater_values[:-1] > 0) & (CWater_values[1:] <= 0))[0]

    # Find the indices where the sign changes from negative to positive
    negative_to_positive_indices = np.where((CWater_values[:-1] < 0) & (CWater_values[1:] >= 0))[0]

    # Extract the positions at which the sign changes
    positions_positive_to_negative = [plotOverLineData.GetPoints().GetPoint(index) for index in positive_to_negative_indices]
    positions_negative_to_positive = [plotOverLineData.GetPoints().GetPoint(index) for index in negative_to_positive_indices]

    # Compute the center of mass as the average position between these two points
    center_of_mass = np.mean([positions_positive_to_negative, positions_negative_to_positive], axis=0)

    # # Calculate the time difference
    # if previous_time is not None:
    #     time_difference = time_step - previous_time
    # else:
    #     time_difference = 0

    # Calculate the velocity
    velocity = (center_of_mass - previous_center_of_mass) / write_interval
    #print(f'Velocity at time {time_step}: {velocity[0][0]:.4f} m/s')
    velocities.append(velocity[0][0])
    
    # Remember the center of mass and time for the next iteration
    previous_center_of_mass = center_of_mass
    previous_time = time_step

    # Extract the first value of the center of mass vector
    first_value_center_of_mass = center_of_mass[0][0]

    # Print the maximum position
    # print(f'max distance traveled by droplet: {first_value_center_of_mass:.4g}, got past the obstacle? {int(first_value_center_of_mass > 0.004)}')
    # print(int(first_value_center_of_mass > 0.004))

if args.We:
    We = rho*max(velocities)*max(velocities)*R0/sigma
    print(We)
if args.L:
    L = Lh/R0
    print(L)
if args.Bo:
    Bo = rho*g*R0*R0/sigma
    print(Bo)
if args.n:
    int(first_value_center_of_mass > 0.004)
    # Print the maximum position
    print(int(first_value_center_of_mass > 0.01))
if args.i:
    # Find the locations where the sign changes from negative to positive
    transitions = np.where((CWater_values[:-1] < 0) & (CWater_values[1:] > 0))[0]
    # Count the number of droplets
    print(len(transitions))


