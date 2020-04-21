#! /usr/bin/env python

from coordinates import CartesianPoint, CartesianVector, ECEF, GeodeticPoint,  \
                        HorizontalVector, LtpFrame, SphericalPoint,            \
                        _units as units


if __name__ == '__main__':
    geodetic = GeodeticPoint(45, 3, units=units.deg)
    frame = LtpFrame(geodetic, orientation='ENU', magnetic=False)
    cartesian = CartesianPoint.new(geodetic, frame)
    spherical = SphericalPoint.new(cartesian, frame=ECEF)
    vz = CartesianVector((0, 0, 1), units=units.m, frame=frame)
    horizontal = HorizontalVector.new(vz)

    print(geodetic)
    print(cartesian)
    print(spherical)
    print(vz)
    print(horizontal)

    geodetic = GeodeticPoint.new(cartesian)
    print(geodetic)

    cartesian = CartesianPoint(((1, 2, 3), (4, 5, 6)), units=units.m)
    cartesian.x = 5 * units.m
    print(cartesian)
    print(type(cartesian.xyz.magnitude))
