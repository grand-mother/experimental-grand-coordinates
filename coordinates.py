from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy

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


    def itransform(self, frame: Frame) -> CartesianCoordinates:
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


    def transform(self, frame: Frame) -> CartesianCoordinates:
        return self.copy().itransform(frame)


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
            new = cls._from_spherical(coordinates)
        elif isinstance(coordinates, GeodeticPoint):
            new = cls._from_geodetic(coordinates)
        else:
            raise NotImplemented(
                f'expected an instance of CartesianCoordinates or Point. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform(frame)

        return new


    @classmethod
    def _from_geodetic(cls, coordinates: Coordinates):
        ecef = turtle.ecef_from_geodetic(
            coordinates.latitude.to(_units.deg).magnitude,
            coordinates.longitude.to(_units.deg).magnitude,
            coordinates.height.to(_units.m).magnitude)
        return cls(Quantity(ecef, _units.m))


class CartesianVector(CartesianCoordinates, Vector):
    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> CartesianVector:
        if isinstance(value, CartesianCoordinates):
            new = coordinates.copy()
        elif isinstance(coordinates, SphericalVector):
            new = cls._from_spherical(coordinates)
        elif isinstance(coordinates, HorizontalVector):
            new = cls._from_horizontal(coordinates)
        else:
            raise NotImplemented(
                f'expected an instance of CartesianCoordinates or Vector. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform(frame)

        return new


    @classmethod
    def _from_horizontal(cls, coordinates: Coordinates):
        ninety = Quantity(90, _units.deg)
        theta = ninety - coordinates.elevation
        phi = ninety - coordinates.azimuth
        ct, st = numpy.cos(theta), numpy.sin(theta)
        r = coordinates.r

        xyz = numpy.empty((theta.size, 3))
        xyz[:, 0] = r * numpy.cos(phi) * st
        xyz[:, 1] = r * numpy.sin(phi) * st
        xyz[:, 2] = r * ct
        quantity = Quantity(xyz, r.units)

        return cls(quantity, self.frame)


@dataclass
class GeodeticPoint(Coordinates, Point):
    latitude: Quantity
    longitude: Quantity
    height: Quantity


    def __init__(self, latitude: Union[float, Quantity, Sequence],
                       longitude: Union[float, Quantity, Sequence],
                       height: Union[float, Quantity, Sequence, None]=None,
                       units: Union[Unit, Sequence[Unit], None]=None):
        if isinstance(units, Unit):
            units = (units, units)

        if not isinstance(latitude, Quantity):
            try:
                latitude = Quantity(latitude, units[0])
            except TypeError:
                raise ValueError('missing units for latitude')

        try:
            l = latitude.size
        except AttributeError:
            l = 0

        if not isinstance(longitude, Quantity):
            try:
                longitude = Quantity(longitude, units[1])
            except TypeError:
                raise ValueError('missing units for longitude')

        try:
            m = longitude.size
        except AttributeError:
            m = 0

        if height is None:
            n = l
            value = numpy.zeros(n) if n > 0 else 0
            height = Quantity(value, units=_units.m)
        else:
            if not isinstance(height, Quantity):
                try:
                    height = Quantity(height, units[2])
                except TypeError:
                    raise ValueError('missing units for height')

            try:
                m = height.size
            except AttributeError:
                m = 0

        if (l != m) or (m != n):
            raise ValueError(
                'longitude, latitude and height must have the same size.')

        self.latitude = latitude
        self.longitude = longitude
        self.height = height


    @classmethod
    def new(cls, coordinates: Coordinates):
        if isinstance(coordinates, GeodeticPoint):
            return coordinates.copy()
        elif isinstance(coordinates, CartesianPoint):
            return cls._from_cartesian()
        elif isinstance(coordinates, SphericalPoint):
            return cls._from_spherical()
        else:
            raise NotImplemented(
                f'expected an instance of GeodeticPoint or Point. '
                f'Got a {type(coordinates)} instead.')


    @classmethod
    def _from_cartesian(cls, coordinates: CartesianCoordinates):
        ecef = coordinates.xyz.flatten(order='C')
        geodetic = turtle.ecef_to_geodetic(ecef)
        latitude = Quantity(geodetic[0], _units.deg)
        longitude = Quantity(geodetic[1], _units.deg)
        height = Quantity(geodetic[2], _units.m)

        return self.__class__(latitude, longitude, height)


    def transform(self, frame: Frame) -> CartesianCoordinates:
        pass


class SphericalCoordinates(Coordinates):
    pass


class SphericalPoint(SphericalCoordinates, Point):
    pass


class SphericalVector(SphericalCoordinates, Vector):
    pass


@dataclass
class LtpFrame(Frame):
    basis: numpy.ndarray
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
