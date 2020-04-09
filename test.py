#! /usr/bin/env python

from coordinates import CartesianPoint, GeodeticPoint, LtpFrame, _units as units


if __name__ == '__main__':
    geodetic = GeodeticPoint(45, 3, units=units.deg)
    frame = LtpFrame(geodetic, orientation='ENU', magnetic=False)
    cartesian = CartesianPoint.new(geodetic, frame)

    print(geodetic)
    print(cartesian)

    cartesian = CartesianPoint(((1, 2, 3), (4, 5, 6)), units=units.m)
    cartesian.x = 5 * units.m
    print(cartesian)
    print(type(cartesian.xyz.magnitude))
