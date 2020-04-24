from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union
import weakref

import numpy
from numpy import ndarray
from pint import UnitRegistry
_units = UnitRegistry()
Quantity, Unit = _units.Quantity, _units.Unit

from _coordinates import ffi, lib


class FrameView:
    def _set_view(self):
        self._basis = numpy.frombuffer(ffi.buffer(self._c[0].basis, 72),
                                       dtype=numpy.float64)
        origin = numpy.frombuffer(ffi.buffer(self._c[0].origin, 24),
                                  dtype=numpy.float64)
        self._origin = Quantity(origin, _units.m)


    @property
    def basis(self):
        return self._basis


    @property
    def origin(self):
        return self._origin


class Frame(FrameView):
    def __init__(self):
        self._c = ffi.new('struct grand_frame [1]')
        self._set_view()


ECEF = FrameView()
ECEF._c = lib.GRAND_ECEF
ECEF._set_view()
ECEF.__class__ = Frame
ECEF.basis.flags.writeable = False
ECEF.origin.flags.writeable = False


class CoordinatesArray:
    '''Proxy for an array of coordinates
    '''
    def __init__(self, size: int=0, type_: int=0, system: int=0,
                 frame: Frame=ECEF):
        ptr = ffi.new('struct grand_coordinates_array* [1]')
        lib.grand_coordinates_array_create(ptr, type_, system, frame._c, size)
        self._cptr = ptr

        self._c = ptr[0]
        self._data = numpy.frombuffer(ffi.buffer(self._c[0].data, 24 * size),
                                      dtype=numpy.float64)
        self._frame = frame

        def destroy():
            lib.grand_coordinates_array_destroy(self._cptr)

        weakref.finalize(self, destroy)


    def resize(self, size: int) -> None:
        lib.grand_coordinates_array_resize(self._cptr, size)

        self._c = self._cptr[0]
        self._data = numpy.frombuffer(ffi.buffer(self._c[0].data, 24 * size),
                                      dtype=numpy.float64)


    @property
    def data(self) -> ndarray:
        return self._data


    @property
    def frame(self) -> Frame:
        return self._frame


    @frame.setter
    def frame(self, frame:Frame) -> None:
        self._c[0].frame = frame._c
        self._frame = frame


    def size(self) -> int:
        return int(self._c[0].size)


class Coordinates(CoordinatesArray):
    def __init__(self, units: Quantity,
                       frame: Frame=ECEF):

        super().__init__(units, 1, frame)


class CartesianView:
    class _CartesianViewData(NamedTuple):
        xyz: Quantity
        x: Quantity
        y: Quantity
        z: Quantity


    def _set_view(self, units):
        xyz = Quantity(numpy.reshape(self._data,
                                     (self._c[0].size, 3)), units)
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        self._view = self._CartesianViewData(xyz, x, y, z)


class CartesianCoordinatesArrayView(CartesianView):
    @property
    def xyz(self) -> Quantity:
        return self._view.xyz


    @property
    def x(self) -> Quantity:
        return self._view.x


    @property
    def y(self) -> Quantity:
        return self._view.y


    @property
    def z(self) -> Quantity:
        return self._view.z


class CartesianCoordinatesArray(CoordinatesArray,
                                CartesianCoordinatesArrayView):
    _system = lib.GRAND_COORDINATES_CARTESIAN
    _type = lib.GRAND_COORDINATES_UNDEFINED_TYPE

    @classmethod
    def new(cls, xyz: Union[ndarray, Sequence],
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None)                               \
                 -> CartesianCoordinatesArray:

        xyz = cls._as_quantity(xyz, units)

        size = xyz.size // 3
        if 3 * size != xyz.size:
            raise ValueError('invalid data size')
        self = cls(xyz.units, size, frame)
        self._view.xyz.magnitude[:] = xyz.magnitude

        return self


    @staticmethod
    def _as_quantity(xyz: Union[Quantity, ndarray, Sequence],
                     units: Union[Quantity, None])                             \
                     -> Quantity:

        if units is not None:
            return Quantity(xyz, units)
        elif not isinstance(xyz, Quantity):
            raise TypeError('missing units')
        else:
            return xyz


    def __init__(self, units: Quantity,
                       size: int=0,
                       frame: Frame=ECEF):

        super().__init__(size, self._type, self._system, frame)
        self._set_view(units)


class CartesianPoints(CartesianCoordinatesArray):
    _type = lib.GRAND_COORDINATES_POINT


class CartesianVectors(CartesianCoordinatesArray):
    _type = lib.GRAND_COORDINATES_VECTOR


class CartesianCoordinatesView(CartesianView):
    @property
    def xyz(self) -> Quantity:
        return self._view.xyz[0]


    @xyz.setter
    def xyz(self, value: Quantity) -> None:
        self._view.xyz[0,:] = value


    @property
    def x(self) -> Quantity:
        return self._view.x[0]


    @x.setter
    def x(self, value: Quantity) -> None:
        self._view.x[0] = value


    @property
    def y(self) -> Quantity:
        return self._view.y[0]


    @y.setter
    def y(self, value: Quantity) -> None:
        self._view.y[0] = value


    @property
    def z(self) -> Quantity:
        return self._view.z[0]


    @z.setter
    def z(self, value: Quantity) -> None:
        self._view.z[0] = value


class CartesianCoordinates(Coordinates,
                           CartesianCoordinatesView,
                           CartesianCoordinatesArray):
    @classmethod
    def new(cls, xyz: Union[Quantity, ndarray, Sequence],
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None)                               \
                 -> CartesianCoordinates:

        xyz = cls._as_quantity(xyz, units)

        if xyz.size != 3:
            raise ValueError('invalid data size')
        self = cls(xyz.units, frame)
        self._view.xyz.magnitude[:] = xyz.magnitude

        return self


class CartesianPoint(CartesianCoordinates):
    _type = lib.GRAND_COORDINATES_POINT


class CartesianVector(CartesianCoordinates):
    _type = lib.GRAND_COORDINATES_VECTOR


class SphericalView:
    class _SphericalViewData(NamedTuple):
        r: Quantity
        theta: Quantity
        phi: Quantity


    def _set_view(self, units: Tuple[Quantity]):
        r = Quantity(self._data[::3], units[0])
        theta = Quantity(self._data[1::3], units[1])
        phi = Quantity(self._data[2::3], units[2])
        self._view = self._SphericalViewData(r, theta, phi)


class SphericalCoordinatesArrayView(SphericalView):
    @property
    def r(self) -> Quantity:
        return self._view.r


    @property
    def theta(self) -> Quantity:
        return self._view.theta


    @property
    def phi(self) -> Quantity:
        return self._view.phi



class SphericalCoordinatesArray(CoordinatesArray,
                                SphericalCoordinatesArrayView):
    _system = lib.GRAND_COORDINATES_SPHERICAL
    _type = lib.GRAND_COORDINATES_UNDEFINED_TYPE

    @classmethod
    def new(cls, r: Union[Quantity, ndarray, Sequence],
                 theta: Union[Quantity, ndarray, Sequence],
                 phi: Union[Quantity, ndarray, Sequence],
                 type_: int=0,
                 frame: Frame=ECEF,
                 units: Optional[Tuple[Quantity]]=None)                        \
                 -> SphericalCoordinatesArray:

        r, theta, phi = cls._as_quantity(r, theta, phi, units)

        self = cls((r.units, theta.units, phi.units), r.size, frame)
        self._view.r.magnitude[:] = r.magnitude
        self._view.theta.magnitude[:] = theta.magnitude
        self._view.phi.magnitude[:] = phi.magnitude

        return self


    @staticmethod
    def _as_quantity(r: ndarray,
                     theta: ndarray,
                     phi: ndarray,
                     units: Union[Tuple[Quantity], None])                      \
                     -> Tuple[Quantity]:

        if units is not None:
            n = len(units)
            if n < 2:
                raise TypeError('missing units')
            r = Quantity(r, units[0])
            theta = Quantity(theta, units[1])
            phi = Quantity(phi, units[-1])
        elif not (isinstance(r, Quantity) and
                  isinstance(theta, Quantity) and
                  isinstance(phi, Quantity)):
            raise TypeError('missing units')

        try:
            r_size = r.size
        except AttributeError:
            r_size = 0

        try:
            theta_size = theta.size
        except AttributeError:
            theta_size = 0

        try:
            phi_size = phi.size
        except AttributeError:
            phi_size = 0

        if (theta_size != r_size) or (phi_size != r_size):
            raise ValueError('invalid data size')

        return r, theta, phi


    def __init__(self, units: Tuple[Quantity],
                       size: int=0,
                       frame: Frame=ECEF):

        super().__init__(size, self._type, self._system, frame)
        self._set_view(units)


class SphericalPoints(SphericalCoordinatesArray):
    _type = lib.GRAND_COORDINATES_POINT


class SphericalVectors(SphericalCoordinatesArray):
    _type = lib.GRAND_COORDINATES_VECTOR


class SphericalCoordinatesView(SphericalView):
    @property
    def r(self) -> Quantity:
        return self._view.r[0]


    @r.setter
    def r(self, value: Quantity) -> None:
        self._view.r[0] = value


    @property
    def theta(self) -> Quantity:
        return self._view.theta[0]


    @theta.setter
    def theta(self, value: Quantity) -> None:
        self._view.theta[0] = value


    @property
    def phi(self) -> Quantity:
        return self._view.phi[0]


    @phi.setter
    def phi(self, value: Quantity) -> None:
        self._view.phi[0] = value


class SphericalCoordinates(Coordinates,
                           SphericalCoordinatesView,
                           SphericalCoordinatesArray):
    @classmethod
    def new(cls, r: Union[Quantity, float],
                 theta: Union[Quantity, float],
                 phi: Union[Quantity, float],
                 frame: Frame=ECEF,
                 units: Optional[Tuple[Quantity]]=None)                        \
                 -> SphericalCoordinates:

        r, theta, phi = cls._as_quantity(r, theta, phi, units)
        try:
            size = r.size
        except AttributeError:
            pass
        else:
            if r.size != 1:
                raise ValueError('invalid data size')
            else:
                r, theta, phi = r[0], theta[0], phi[0]

        self = cls((r.units, theta.units, phi.units), frame)
        self._view.r.magnitude[0] = r.magnitude
        self._view.theta.magnitude[0] = theta.magnitude
        self._view.phi.magnitude[0] = phi.magnitude

        return self


class SphericalPoint(SphericalCoordinates):
    _type = lib.GRAND_COORDINATES_POINT


class SphericalVector(SphericalCoordinates):
    _type = lib.GRAND_COORDINATES_VECTOR
