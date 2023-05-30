res = 30;

xmax = 3.22;
ymax = 1.0;
zmax = 1.0;

bxmin = 0.67;
bxmax = 0.83;

bymin = -0.2;
bymax = 0.2;

bzmin = 0;
bzmax = 0.16;

bl = (bxmin+bxmax)/2.0;
bw = (bymin+bymax)/2.0;
bh = (bzmin+bzmax)/2.0;

difference () {
    translate([0, -ymax/2.0, 0]) cube([xmax, ymax, zmax]);
    translate([bxmin, bymin, 0]) cube([bxmax-bxmin, bymax-bymin, bzmax-bzmin]);
}