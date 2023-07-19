// Run with openscad pitzDaily.scad -o pitzDaily.stl -D 'nCtrlPnts=1;...'

// 0.0 Tell the used version
echo(version=version());
assert(version_num()>=20190500, "Requires OpenSCAD version 2019.05 or later.");

// Needs https://raw.githubusercontent.com/BelfrySCAD/BOSL2/master/utility.scad
include <./utility.scad>
// Needs https://raw.githubusercontent.com/BelfrySCAD/BOSL2/master/math.scad
include <./math.scad>

// For parameter variation, supply variables in section 0.0 and/or 0.4

// 0.0 Overridable params, these are exposed to parameter variation

// Number of points to generate on the lower patch curve
nCurvePnts = 15;
// Number of control points for the curve
// including both ends. This better be an even number
nCtrlPnts = 4;
// Domain length
L = 0.29;

// 0.1 Needed fixed params

// PitzDaily original points
upperPnts = [
    [-0.0206, 0.0254 , 0], // inlet0
    [-0.0206, 0      , 0], // inlet1
    [0      , 0      , 0], // lowerWallB0
    [0.29   , -0.0166, 0], // lowerWallB1
    [0.29   , 0.0166 , 0], // outlet
    [0.206  , 0.0254 , 0]  // upperWall
];

// 0.2 Meta functions

// The way we parametrically generate control points is simply
// by translating existing ones by an angle and a distance
function newCtrlPoint(point, angle, r) = point + [r*cos(angle),r*sin(angle),0];

// Bezier coords along the curve
// Implementing B(s) = sum((n j) s^j (1-s)^(n-j).P_j)
function bezierCoord(s, ctrlPnts) = [
    sum([
        for (j=[0:len(ctrlPnts)-1]) binomial(len(ctrlPnts)-1)[j] * pow(s,j) * pow(1.0-s, len(ctrlPnts)-1-j) * ctrlPnts[j][0],
    ]),
    sum([
        for (j=[0:len(ctrlPnts)-1]) binomial(len(ctrlPnts)-1)[j] * pow(s,j) * pow(1.0-s, len(ctrlPnts)-1-j) * ctrlPnts[j][1],
    ]),
    sum([
        for (j=[0:len(ctrlPnts)-1]) binomial(len(ctrlPnts)-1)[j] * pow(s,j) * pow(1.0-s, len(ctrlPnts)-1-j) * ctrlPnts[j][2],
    ])
];

// 0.3 Overridable lists, can be modified from CMD

// Angles and rations of domain length to put control points at
angles = [-90, -150];
radius_ratios = [for (j=[0:nCtrlPnts-3]) 0.1];

echo(angles=angles);
echo(radius=radius_ratios);

// 0.4 Fixed lists

radius = [for (j=[0:nCtrlPnts-3]) radius_ratios[j]*L];
origins = [for (j=[0:nCtrlPnts-3]) upperPnts[j+2]];

// Control points for a bezier curve in XZ plane
ctrlPnts = [
    upperPnts[2],
    for(j=[0:nCtrlPnts-3]) newCtrlPoint(origins[j], angles[j], radius[j]),
    upperPnts[3]
];

// Generate lower wall curve
lowerWall = [
    for (i=[0:nCurvePnts]) bezierCoord(i/nCurvePnts, ctrlPnts),
];

// Genererate the "bottom" patch
patch = [
    upperPnts[0],
    upperPnts[1],
    for (i = [0 : len(lowerWall)-1]) lowerWall[i],
    upperPnts[4],
    upperPnts[5],
];

// Extrude following Z and produce the geometry
linear_extrude(0.001)
{
    polygon(points=[for (i = [0 : len(patch)-1]) [patch[i][0],patch[i][1]]]);
};
