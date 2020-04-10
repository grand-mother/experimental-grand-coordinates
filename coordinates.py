from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy
from numpy import ndarray

# from . import units
from grand.libs import turtle
from pint import UnitRegistry
_units = UnitRegistry()
Quantity, Unit = _units.Quantity, _units.Unit


__all__ = ['CartesianCoordinates', 'CartesianPoint', 'CartesianVector',
           'Coordinates', 'EcefFrame', 'ECEF', 'Frame', 'GeodeticPoint',
           'HorizontalVector', 'LtpFrame', 'Point', 'SphericalCoordinates',
           'SphericalPoint', 'SphericalVector', 'Vector']


_HAS_GEOMAGNET: bool = False # Differ the import of the geomagnet module
                             # in order to avoid circular references


def _cartesian_to_geodetic(xyz: ndarray) -> Tuple[Union[float, ndarray]]:
    '''Transform Cartesian coordinates to geodetic ones
    '''
    ecef = xyz.flatten(order='C')
    return turtle.ecef_to_geodetic(ecef)


_cartesian_to_geodetic_u = _units.wraps(
    (_units.deg, _units.deg, _units.m), _units.m)(_cartesian_to_geodetic)


def _geodetic_to_cartesian(latitude: Union[float, ndarray],
                           longitude: Union[float, ndarray],
                           height: Union[float, ndarray])                      \
                           -> Quantity:
    '''Transform geodetic coordinates to Cartesian ones
    '''
    return turtle.ecef_from_geodetic(latitude, longitude, height)


_geodetic_to_cartesian_u = _units.wraps(
    _units.m, (_units.deg, _units.deg, _units.m))(_geodetic_to_cartesian)


def _cartesian_to_spherical(xyz: ndarray) -> Tuple[ndarray]:

    x, y, z = xyz
    rho2 = x**2 + y**2
    rho = numpy.sqrt(rho2)
    theta = numpy.arctan2(rho, z)

    if theta == 0:
        phi = 0
    else:
        phi = numpy.arctan2(y, x)

    r = numpy.sqrt(rho2 + z**2)

    return theta, phi, r


_cartesian_to_spherical_u = _units.wraps(
    (_units.rad, _units.rad, '=A'), '=A')(_cartesian_to_spherical)


def _spherical_to_cartesian(theta: Union[float, ndarray],
                            phi: Union[float, ndarray],
                            r: Union[float, ndarray])                          \
                            -> Quantity:

    ct, st = numpy.cos(theta), numpy.sin(theta)

    xyz = numpy.empty((theta.size, 3))
    xyz[:, 0] = r * numpy.cos(phi) * st
    xyz[:, 1] = r * numpy.sin(phi) * st
    xyz[:, 2] = r * ct

    return xyz


_spherical_to_cartesian_u = _units.wraps(
    '=A', (_units.rad, _units.rad, '=A'))(_spherical_to_cartesian)


class Coordinates:
    '''Abstract type for coordinates
    '''
    pass


class Frame:
    '''Abstract type for reference frames
    '''
    pass


class Point:
    '''Generic point attribute
    '''
    pass


class Vector:
    '''Generic vector attribute
    '''
    pass


class EcefFrame(Frame):
    '''Geocentric frame (Earth-Centered, Earth-Fixed)
    '''
    def __repr__(self):
        return 'ECEF'


ECEF = EcefFrame()


@dataclass
class CartesianCoordinates(Coordinates):
    xyz: Quantity
    frame: Frame


    def __init__(self, xyz: Union[Quantity, Sequence],
                       frame: Frame=ECEF,
                       units: Quantity=None):
        if units is not None:
            xyz = Quantity(xyz, units)
        self.xyz = xyz
        self.frame = frame


    def copy(self):
        return self.__class__(self.xyz.copy(), self.frame)


    def itransform_to(self, frame: Frame) -> CartesianCoordinates:
        if self.frame is None:
            raise ValueError('missing frame for coordinates')
        elif frame is self.frame:
            pass
        elif self.frame is ECEF:
            if isinstance(self, Point):
                t = frame.origin
                self.x -= t.x
                self.y -= t.y
                self.z -= t.z
            self.xyz[:] = numpy.dot(frame.basis.T, self.xyz)
        elif frame is ECEF:
            self.xyz[:] = numpy.dot(frame.basis, self.xyz)
            if isinstance(self, Point):
                t = frame.origin
                self.x += t.x
                self.y += t.y
                self.z += t.z
        else:
            self.xyz[:] = numpy.dot(self.frame.basis, self.xyz)
            if isinstance(self, Point):
                t = self.frame.origin - frame.origin
                self.x += t.x
                self.y += t.y
                self.z += t.z
            self.xyz[:] = numpy.dot(frame.basis.T, self.xyz)

        self.frame = frame
        return self


    def transform_to(self, frame: Frame) -> CartesianCoordinates:
        return self.copy().itransform_to(frame)


    @property
    def x(self) -> Quantity:
        return self.xyz[:, 0] if self.xyz.size > 3 else self.xyz[0]


    @x.setter
    def x(self, value: Quantity) -> None:
        if self.xyz.size > 3:
            self.xyz[:, 0] = value
        else:
            self.xyz[0] = value


    @property
    def y(self) -> Quantity:
        return self.xyz[:, 1] if self.xyz.size > 3 else self.xyz[1]


    @y.setter
    def y(self, value: Quantity) -> None:
        if self.xyz.size > 3:
            self.xyz[:, 1] = value
        else:
            self.xyz[1] = value


    @property
    def z(self) -> Quantity:
        return self.xyz[:, 2] if self.xyz.size > 3 else self.xyz[2]


    @z.setter
    def z(self, value: Quantity) -> None:
        if self.xyz.size > 3:
            self.xyz[:, 2] = value
        else:
            self.xyz[2] = value


class CartesianPoint(CartesianCoordinates, Point):
    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> CartesianPoint:
        if isinstance(coordinates, CartesianCoordinates):
            new = coordinates.copy()
        elif isinstance(coordinates, SphericalPoint):
            xyz = _spherical_to_cartesian_u(coordinates.theta, coordinates.phi,
                                            coordinates.r)
            new = cls(xyz, frame=coordinates.frame)
        elif isinstance(coordinates, GeodeticPoint):
            xyz = _geodetic_to_cartesian_u(coordinates.latitude,
                                           coordinates.longitude,
                                           coordinates.height)
            new = cls(xyz, frame=ECEF)
        else:
            raise NotImplemented(
                f'expected an instance of CartesianCoordinates or Point. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform_to(frame)

        return new


class CartesianVector(CartesianCoordinates, Vector):
    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> CartesianVector:
        if isinstance(value, CartesianCoordinates):
            new = coordinates.copy()
        elif isinstance(coordinates, SphericalVector):
            xyz = _spherical_to_cartesian_u(coordinates.theta, coordinates.phi,
                                            coordinates.r)
            new = cls(xyz, frame=coordinates.frame)
        elif isinstance(coordinates, HorizontalVector):
            xyz = _horizontal_to_cartesian_u(coordinates.latitude,
                                             coordinates.longitude,
                                             coordinate.height)
            new = cls(xyz, frame=coordinates.frame)
        else:
            raise NotImplemented(
                f'expected an instance of CartesianCoordinates or Vector. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform_to(frame)

        return new


def _initialise_angular_coordinates(labels: Sequence[str],
                                    variables: Sequence,
                                    units: Union[Unit, Sequence[Unit], None])  \
                                    -> Sequence:

    if isinstance(units, Unit):
        units = (units, units)

    results, size = [None, None, None], None
    for i, var in enumerate(variables):
        if var is None:
            value = numpy.zeros(n) if n > 0 else 0
            var = Quantity(value, units=_units.m)
        else:
            if not isinstance(var, Quantity):
                try:
                    var = Quantity(var, units[i])
                except TypeError:
                    raise ValueError(f'missing units for {labels[i]}')

            try:
                n = var.size
            except AttributeError:
                n = 0
            if size is not None and n != size:
                raise ValueError(
                    '{labels[i - 1]}, {labels[i]} must have the same size.')
            size = n
        results[i] = var

    return results


@dataclass
class GeodeticPoint(Coordinates, Point):
    latitude: Quantity
    longitude: Quantity
    height: Quantity


    def __init__(self, latitude: Union[float, Quantity, Sequence],
                       longitude: Union[float, Quantity, Sequence],
                       height: Union[float, Quantity, Sequence, None]=None,
                       units: Union[Unit, Sequence[Unit], None]=None):

        self.latitude, self.longitude, self.height =                           \
            _initialise_angular_coordinates(
            ('latitude', 'longitude', 'height'),
             (latitude, longitude, height),
             units)


    @classmethod
    def new(cls, coordinates: Coordinates):
        if isinstance(coordinates, GeodeticPoint):
            return coordinates.copy()
        elif isinstance(coordinates, CartesianPoint):
            if coordinates.frame != ECEF:
                coordinates = coordinates.transform_to(ECEF)
            lat, lon, height = _cartesian_to_geodetic_u(coordinates.xyz)
            return cls(lat, lon, height)
        elif isinstance(coordinates, SphericalPoint):
            if coordinates.frame != ECEF:
                coordinates = coordinates.transform_to(ECEF)
            lat, lon, height = _spherical_to_geodetic_u(coordinates.theta,
                                                        coordinates.phi,
                                                        coordinates.r)
            return cls(lat, lon, height)
        else:
            raise NotImplemented(
                f'expected an instance of GeodeticPoint or Point. '
                f'Got a {type(coordinates)} instead.')


    def copy(self):
        return copy.deepcopy(self)


@dataclass
class SphericalCoordinates(Coordinates):
    theta: Quantity
    phi: Quantity
    r: Quantity


    def __init__(self, theta: Union[float, Quantity, Sequence],
                       phi: Union[float, Quantity, Sequence],
                       r: Union[float, Quantity, Sequence, None]=None,
                       frame: Frame=ECEF,
                       units: Union[Unit, Sequence[Unit], None]=None):

        self.theta, self.phi, self.r = _initialise_angular_coordinates(
            ('latitude', 'longitude', 'height'),
            (latitude, longitude, height),
            units)
        self.frame = frame


    def copy(self):
        new = copy.copy(self)
        new.theta = self.theta.copy()
        new.phi = self.phi.copy()
        new.r = self.r.copy()
        return new


    def itransform(self, frame: Frame) -> SphericalCoordinates:
        new = self.transform(frame)
        self.theta[:] = new.theta
        self.phi[:] = new.phi
        self.r[:] = new.r
        self.frame = frame
        return self


    def transform(self, frame: Frame) -> SphericalCoordinates:
        xyz = _spherical_to_cartesian(self.theta, self.phi, self.r)
        _cartesian_transform(xyz, frame)
        theta, phi, r = _cartesian_to_spherical(xyz)

        return self.__cls__(theta, phi, r, frame)


class SphericalPoint(SphericalCoordinates, Point):
    pass


class SphericalVector(SphericalCoordinates, Vector):
    pass


@dataclass
class LtpFrame(Frame):
    basis: ndarray
    origin: CartesianPoint


    def __init__(self, origin: Coordinates,
                       orientation: Union[Sequence[str], str, None]=None,
                       magnetic: Optional[bool]=None,
                       declination: Optional[Quantity]=None,
                       rotation: Optional[Rotation]=None):

        if isinstance(origin, GeodeticPoint):
            geodetic_origin = origin
        else:
            geodetic_origin = GeodeticPoint.new(origin)
        latitude = geodetic_origin.latitude.to(_units.deg).magnitude,
        longitude = geodetic_origin.longitude.to(_units.deg).magnitude

        if magnetic and declination is None:
            # Compute the magnetic declination
            ecef = ECEF(itrs.x, itrs.y, itrs.z, obstime=self._obstime)

            global _HAS_GEOMAGNET
            if not _HAS_GEOMAGNET:
                from .geomagnet import field as _geomagnetic_field
                _HAS_GEOMAGNET = True

            field = _geomagnetic_field(geodetic_origin)
            if not isinstance(field, HorizontalVector):
                field = HorizontalVector.new(field)
            declination = field.azimuth.to(_units.deg).magnitude

        if declination is None:
            azimuth0 = 0
        else:
            azimuth0 = declination

        def vector(name):
            tag = name[0].upper()
            if tag == 'E':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   90 + azimuth0, 0)
            elif tag == 'W':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   270 + azimuth0, 0)
            elif tag == 'N':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   azimuth0,  0)
            elif tag == 'S':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   180 + azimuth0,  0)
            elif tag == 'U':
                return turtle.ecef_from_horizontal(latitude, longitude, 0, 90)
            elif tag == 'D':
                return turtle.ecef_from_horizontal(latitude, longitude, 0, -90)
            else:
                raise ValueError(f'Invalid frame orientation `{name}`')

        ux = vector(orientation[0])
        uy = vector(orientation[1])
        uz = vector(orientation[2])

        self.basis = numpy.column_stack((ux, uy, uz))
        self.origin = CartesianPoint.new(origin)

        if rotation is not None:
            self.basis = rotation.apply(self.basis, inverse=True)


    def rotated(self, rotation: Rotation) -> LtpFrame:
        '''Get a rotated version of this frame.
        '''
        r = rotation if self.rotation is None else rotation * self.rotation
        frame = self.copy()
        frame.basis = rotation.apply(frame.basis, inverse=True)
        return frame
