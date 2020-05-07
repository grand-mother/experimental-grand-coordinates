#! /usr/bin/env python

from ccoordinates import CartesianPoint, CartesianPoints, ECEF, Frame,         \
                         SphericalPoint, SphericalPoints, _units as u


print(ECEF.basis, ECEF.origin)

c = CartesianPoints.new(((1, 2, 3), (4, 5, 6)), units=u.m)
c.xyz[0,0] = 1 * u.mm
c.y[1] = 1 * u.cm
print(c.data, c.xyz, c.x, c.y, c.z)

c = CartesianPoint.new((1, 2, 3), units=u.cm)
c.y = 1 * u.km
print(c.xyz, c.x, c.y, c.z)

s = SphericalPoints.new((1, 2), (0, 90), (45, 60), units=(u.m, u.deg))
s.r[1] = 1 * u.cm
print(s.r, s.theta, s.phi)

s = SphericalPoint.new(1 * u.m, 90 * u.deg, 60 * u.deg)
s.r = 1 * u.cm
print(s.r, s.theta, s.phi)
