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


def _cartesian_to_geodetic_r(xyz: ndarray) -> Tuple[Union[float, ndarray]]:
    '''Transform Cartesian coordinates to geodetic ones (raw)
    '''
    ecef = xyz.flatten(order='C')
    return turtle.ecef_to_geodetic(ecef)


_cartesian_to_geodetic = _units.wraps(
    (_units.deg, _units.deg, _units.m), _units.m)(_cartesian_to_geodetic_r)


def _geodetic_to_cartesian_r(latitude: Union[float, ndarray],
                             longitude: Union[float, ndarray],
                             height: Union[float, ndarray])                    \
                             -> ndarray:
    '''Transform geodetic coordinates to Cartesian ones (raw)
    '''
    return turtle.ecef_from_geodetic(latitude, longitude, height)


_geodetic_to_cartesian = _units.wraps(
    _units.m, (_units.deg, _units.deg, _units.m))(_geodetic_to_cartesian_r)


def _cartesian_to_spherical_r(xyz: ndarray) -> Tuple[ndarray]:
    '''Transform Cartesian coordinates to spherical ones (raw)
    '''
    x, y, z = xyz
    rho2 = x**2 + y**2
    rho = numpy.sqrt(rho2)
    theta = numpy.arctan2(rho, z)

    if theta == 0:
        phi = numpy.zeros(theta.shape)
    else:
        phi = numpy.arctan2(y, x)

    r = numpy.sqrt(rho2 + z**2)

    return theta, phi, r


_cartesian_to_spherical = _units.wraps(
    (_units.rad, _units.rad, '=A'), '=A')(_cartesian_to_spherical_r)


def _spherical_to_cartesian_r(theta: Union[float, ndarray],
                              phi: Union[float, ndarray],
                              r: Union[float, ndarray])                        \
                              -> ndarray:
    '''Transform spherical coordinates to Cartesian ones (raw)
    '''
    ct, st = numpy.cos(theta), numpy.sin(theta)
    if isinstance(theta, ndarray) and theta.size > 1:
        xyz = numpy.empty((theta.size, 3))
        x, y, z = xyz
    else:
        xyz = numpy.empty(3)
        x, y, z = xyz[0:1], xyz[1:2], xyz[2:3] # Force getting views instead
                                               # of elements

    x[:] = r * numpy.cos(phi) * st
    y[:] = r * numpy.sin(phi) * st
    z[:] = r * ct

    return xyz


_spherical_to_cartesian = _units.wraps(
    '=A', (_units.rad, _units.rad, '=A'))(_spherical_to_cartesian_r)


def _spherical_to_geodetic_r(theta: Union[float, ndarray],
                             phi: Union[float, ndarray],
                             r: Union[float, ndarray])                         \
                             -> Tuple[Union[float, ndarray]]:
    '''Transform spherical coordinates to geodetic ones (raw)
    '''
    xyz = _spherical_to_cartesian_r(theta, phi, r)
    return _cartesian_to_geodetic_r(xyz)


_spherical_to_geodetic = _units.wraps(
    (_units.deg, _units.deg, _units.m),
    (_units.rad, _units.rad, _units.m))(_spherical_to_geodetic_r)


def _geodetic_to_spherical_r(latitude: Union[float, ndarray],
                             longitude: Union[float, ndarray],
                             height: Union[float, ndarray])                    \
                             -> Tuple[Union[float, ndarray]]:
    '''Transform geodetic coordinates to spherical ones (raw)
    '''
    xyz = _geodetic_to_cartesian_r(latitude, longitude, height)
    return _cartesian_to_spherical_r(xyz)


_geodetic_to_spherical = _units.wraps(
    (_units.rad, _units.rad, _units.m),
    (_units.deg, _units.deg, _units.m))(_geodetic_to_spherical_r)


def _spherical_to_horizontal_r(theta: Union[float, ndarray],
                               phi: Union[float, ndarray],
                               r: Union[float, ndarray])                       \
                               -> Tuple[Union[float, ndarray]]:
    '''Transform spherical coordinates to horizontal ones (raw)
    '''
    return 0.5 * numpy.pi - phi, 0.5 * numpy.pi - theta, r


_spherical_to_horizontal = _units.wraps(
    (_units.rad, _units.rad, '=A'),
    (_units.rad, _units.rad, '=A'))(_spherical_to_horizontal_r)


def _horizontal_to_spherical_r(azimuth: Union[float, ndarray],
                               elevation: Union[float, ndarray],
                               norm: Union[float, ndarray])                    \
                               -> Tuple[Union[float, ndarray]]:
    '''Transform horizontal coordinates to spherical ones (raw)
    '''
    return 0.5 * numpy.pi - elevation, 0.5 * numpy.pi - azimuth, norm


_horizontal_to_spherical = _units.wraps(
    (_units.rad, _units.rad, '=A'),
    (_units.rad, _units.rad, '=A'))(_horizontal_to_spherical_r)


def _cartesian_to_horizontal_r(xyz: ndarray) -> Tuple[Union[float, ndarray]]:
    '''Transform Cartesian coordinates to horizontal ones (raw)
    '''
    theta, phi, r = _cartesian_to_spherical_r(xyz)
    return _spherical_to_horizontal_r(theta, phi, r)


_cartesian_to_horizontal = _units.wraps(
    (_units.rad, _units.rad, '=A'), '=A')(_cartesian_to_horizontal_r)


def _horizontal_to_cartesian_r(azimuth: Union[float, ndarray],
                               elevation: Union[float, ndarray],
                               r: Union[float, ndarray])                       \
                               -> ndarray:
    '''Transform horizontal coordinates to Cartesian ones (raw)
    '''
    theta, phi, r = _horizontal_to_spherical_r(azimuth, elevation, r)
    return _spherical_to_cartesian_r(theta, phi, r)


_horizontal_to_cartesian = _units.wraps(
    '=A', (_units.rad, _units.rad, '=A'))(_horizontal_to_cartesian_r)


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


def _cartesian_itransform_to_r(xyz: ndarray, units, initial_frame: Frame,
                                                    final_frame: Frame):
    '''Transform the reference frame of Cartesian coordinates
    '''
    if xyz.size > 3:
        x, y, z = xyz
    else:
        x, y, z = xyz[0:1], xyz[1:2], xyz[2:3]

    if initial_frame is ECEF:
        if units is not None:
            t = final_frame.origin.xyz.to(units).magnitude
            x -= t[0]
            y -= t[1]
            z -= t[2]
        xyz[:] = numpy.dot(final_frame.basis.T, xyz)
    elif final_frame is ECEF:
        xyz[:] = numpy.dot(initial_frame.basis, xyz)
        if units is not None:
            t = initial_frame.origin.xyz.to(units).magnitude
            x += t[0]
            y += t[1]
            z += t[2]
    else:
        xyz[:] = numpy.dot(initial_frame.basis, xyz)
        if units is not None:
            t = initial_frame.origin.xyz - final_frame.origin.xyz
            t.ito(units)
            t = t.magnitude
            x += t[0]
            y += t[1]
            z += t[2]
        xyz[:] = numpy.dot(final_frame.basis.T, xyz)


@dataclass
class CartesianCoordinates(Coordinates):
    xyz: Quantity
    frame: Frame


    def __init__(self, xyz: Union[Quantity, Sequence],
                       frame: Frame=ECEF,
                       units: Optional[Quantity]=None):
        if units is not None:
            xyz = Quantity(xyz, units)
        self.xyz = xyz
        self.frame = frame


    def __add__(self, coordinates: Coordinates):
        pass # XXX Here I am


    def copy(self):
        return self.__class__(self.xyz.copy(), self.frame)


    def itransform_to(self, frame: Frame) -> CartesianCoordinates:
        if self.frame is None:
            raise ValueError('missing frame for coordinates')
        elif frame is self.frame:
            return self
        else:
            units = self.xyz.units if isinstance(self, Point) else None
            _cartesian_itransform_to_r(self.xyz.magnitude, units, self.frame,
                                                                  frame)
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
            xyz = _spherical_to_cartesian(coordinates.theta, coordinates.phi,
                                            coordinates.r)
            new = cls(xyz, frame=coordinates.frame)
        elif isinstance(coordinates, GeodeticPoint):
            xyz = _geodetic_to_cartesian(coordinates.latitude,
                                         coordinates.longitude,
                                         coordinates.height)
            new = cls(xyz, frame=ECEF)
        else:
            raise NotImplementedError(
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
            xyz = _spherical_to_cartesian(coordinates.theta, coordinates.phi,
                                          coordinates.r)
            new = cls(xyz, frame=coordinates.frame)
        elif isinstance(coordinates, HorizontalVector):
            xyz = _horizontal_to_cartesian(coordinates.latitude,
                                           coordinates.longitude,
                                           coordinate.height)
            new = cls(xyz, frame=coordinates.frame)
        else:
            raise NotImplementedError(
                f'expected an instance of CartesianCoordinates or Vector. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform_to(frame)

        return new


def _initialise_angular_coordinates(labels: Sequence[str],
                                    variables: Sequence,
                                    units: Union[Unit, Sequence[Unit], None],
                                    default: float=0)  \
                                    -> Sequence:

    if isinstance(units, Unit):
        units = (units, units)

    results, size = [None, None, None], None
    for i, var in enumerate(variables):
        if var is None:
            value = numpy.full(n, default) if n > 0 else default
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
                    f'{labels[i - 1]} and {labels[i]} must have the same size.')
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
            lat, lon, height = _cartesian_to_geodetic(coordinates.xyz)
            return cls(lat, lon, height)
        elif isinstance(coordinates, SphericalPoint):
            if coordinates.frame != ECEF:
                coordinates = coordinates.transform_to(ECEF)
            lat, lon, height = _spherical_to_geodetic(coordinates.theta,
                                                        coordinates.phi,
                                                        coordinates.r)
            return cls(lat, lon, height)
        else:
            raise NotImplementedError(
                f'expected an instance of GeodeticPoint or Point. '
                f'Got a {type(coordinates)} instead.')


    def copy(self):
        return copy.deepcopy(self)


@dataclass
class SphericalCoordinates(Coordinates):
    theta: Quantity
    phi: Quantity
    r: Quantity
    frame: Frame


    def __init__(self, theta: Union[float, Quantity, Sequence],
                       phi: Union[float, Quantity, Sequence],
                       r: Union[float, Quantity, Sequence, None]=None,
                       frame: Frame=ECEF,
                       units: Union[Unit, Sequence[Unit], None]=None):

        self.theta, self.phi, self.r = _initialise_angular_coordinates(
            ('theta', 'phi', 'r'),
            (theta, phi, r),
            units,
            1)
        self.frame = frame


    def copy(self):
        new = copy.copy(self)
        new.theta = self.theta.copy()
        new.phi = self.phi.copy()
        new.r = self.r.copy()
        new.frame = self.frame
        return new


    def itransform_to(self, frame: Frame) -> SphericalCoordinates:
        if self.frame is None:
            raise ValueError('missing frame for coordinates')
        elif frame is self.frame:
            return self
        else:
            xyz = _spherical_to_cartesian_r(
                self.theta.to(_units.rad).magnitude,
                self.phi.to(_units.rad).magnitude,
                self.r.magnitude)
            units = self.r.units if isinstance(self, Point) else None
            _cartesian_itransform_to_r(xyz, units, self.frame, frame)
            theta, phi, r = _cartesian_to_spherical_r(xyz)

            theta = _units.Quantity(theta, _units.rad)
            phi = _units.Quantity(phi, _units.rad)
            r = _units.Quantity(r, self.r.units)
            try:
                self.theta[:] = theta
                self.phi[:] = phi
                self.r[:] = r
            except TypeError:
                self.theta = theta
                self.phi = phi
                self.r = r
            self.frame = frame
            return self


    def transform_to(self, frame: Frame) -> SphericalCoordinates:
        return self.copy().itransform_to(frame)


class SphericalPoint(SphericalCoordinates, Point):
    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> SphericalPoint:
        if isinstance(coordinates, SphericalCoordinates):
            new = coordinates.copy()
        elif isinstance(coordinates, CartesianPoint):
            theta, phi, r = _cartesian_to_spherical(coordinates.xyz)
            new = cls(theta, phi, r, frame=coordinates.frame)
        elif isinstance(coordinates, GeodeticPoint):
            theta, phi, r = _geodetic_to_spherical(coordinates.latitude,
                                                   coordinates.longitude,
                                                   coordinates.height)
            new = cls(theta, phi, r, frame=ECEF)
        else:
            raise NotImplementedError(
                f'expected an instance of SphericalCoordinates or Point. '
                f'Got a {type(coordinates)} instead.')

        if frame: # XXX optimize this
            new.itransform_to(frame)

        return new


class SphericalVector(SphericalCoordinates, Vector):
    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> SphericalVector:
        if isinstance(value, SphericalCoordinates):
            new = coordinates.copy()
        elif isinstance(coordinates, CartesianVector):
            theta, phi, r = _cartesian_to_spherical(coordinates.xyz)
            new = cls(theta, phi, r, frame=coordinates.frame)
        elif isinstance(coordinates, HorizontalVector):
            theta, phi, r = _horizontal_to_spherical(coordinates.azimuth,
                                                     coordinates.elevation,
                                                     coordinate.r)
            new = cls(theta, phi, r, frame=coordinates.frame)
        else:
            raise NotImplementedError(
                f'expected an instance of SphericalCoordinates or Vector. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform_to(frame)

        return new


@dataclass
class HorizontalVector(Coordinates, Vector):
    azimuth: Quantity
    elevation: Quantity
    norm: Quantity
    frame: Frame


    def __init__(self, azimuth: Union[float, Quantity, Sequence],
                       elevation: Union[float, Quantity, Sequence],
                       norm: Union[float, Quantity, Sequence, None]=None,
                       frame: Frame=ECEF,
                       units: Union[Unit, Sequence[Unit], None]=None):

        self.azimuth, self.elevation, self.norm =                              \
            _initialise_angular_coordinates(
            ('azimuth', 'elevation', 'norm'),
             (azimuth, elevation, norm),
             units,
             1)
        self.frame = frame


    def copy(self):
        new = copy.copy(self)
        new.azimuth = self.azimuth.copy()
        new.elevation = self.elevation.copy()
        new.norm = self.norm.copy()
        new.frame = self.frame
        return new


    def itransform_to(self, frame: Frame) -> HorizontalVector:
        if self.frame is None:
            raise ValueError('missing frame for coordinates')
        elif frame is self.frame:
            return self
        else:
            xyz = _horizontal_to_cartesian_r(
                self.azimuth.to(_units.rad).magnitude,
                self.elevation.to(_units.rad).magnitude,
                self.norm.magnitude)
            _cartesian_itransform_to_r(xyz, None, self.frame, frame)
            azimuth, elevation, norm = _cartesian_to_horizontal_r(xyz)

            azimuth = _units.Quantity(azimuth, _units.rad)
            elevation = _units.Quantity(elevation, _units.rad)
            norm = _units.Quantity(norm, self.norm.units)
            try:
                self.azimuth[:] = azimuth
                self.elevation[:] = elevation
                self.norm[:] = norm
            except TypeError:
                self.azimuth = azimuth
                self.elevation = elevation
                self.norm = norm
            self.frame = frame
            return self


    def transform_to(self, frame: Frame) -> HorizontalVector:
        return self.copy().itransform_to(frame)


    @classmethod
    def new(cls, coordinates: Coordinates,
                 frame: Optional[Frame]=None) -> HorizontalVector:
        if isinstance(coordinates, HorizontalVector):
            new = coordinates.copy()
        elif isinstance(coordinates, CartesianVector):
            azimuth, elevation, norm = _cartesian_to_horizontal(coordinates.xyz)
            new = cls(azimuth, elevation, norm, frame=coordinates.frame)
        elif isinstance(coordinates, SphericalVector):
            azimuth, elevation, norm =                                         \
                _spherical_to_horizontal(coordinates.theta,
                                         coordinates.phi,
                                         coordinate.r)
            new = cls(azimuth, elevation, norm, frame=coordinates.frame)
        else:
            raise NotImplementedError(
                f'expected an instance of Vector. '
                f'Got a {type(coordinates)} instead.')

        if frame:
            new.itransform_to(frame)

        return new


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
