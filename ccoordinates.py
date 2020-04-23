from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union
import weakref

import numpy
from numpy import ndarray
from pint import UnitRegistry
_units = UnitRegistry()
Quantity, Unit = _units.Quantity, _units.Unit

from _coordinates import ffi, lib


class Frame:
    def __init__(self):
        self._c = ffi.new('struct grand_frame [1]')


ECEF = Frame() # XXX move to C


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


class _CartesianViews(NamedTuple):
    xyz: Quantity
    x: Quantity
    y: Quantity
    z: Quantity


class CartesianCoordinatesArray(CoordinatesArray):
    _system = lib.GRAND_COORDINATES_CARTESIAN

    @classmethod
    def new(cls, xyz: Union[ndarray, Sequence],
                 type_: int=0,
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None)                               \
                 -> CartesianCoordinatesArray:

        xyz = cls._as_quantity(xyz, units)

        size = xyz.size // 3
        if 3 * size != xyz.size:
            raise ValueError('invalid data size')
        self = cls(xyz.units, size, type_, frame)
        self._views.xyz.magnitude[:] = xyz.magnitude

        return self


    @staticmethod
    def _as_quantity(xyz: Union[Quantity, ndarray, Sequence],
                     units: Union[Quantity, None])                              \
                     -> Quantity:

        if units is not None:
            return Quantity(xyz, units)
        elif not isinstance(xyz, Quantity):
            raise TypeError('missing units')
        else:
            return xyz


    def __init__(self, units: Quantity,
                       size: int=0,
                       type_: int=0, 
                       frame: Frame=ECEF):

        super().__init__(size, type_, self._system, frame)
        self._set_views(units)


    def _set_views(self, units):
        xyz = Quantity(numpy.reshape(self._data,
                                     (self._c[0].size, 3)), units)
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        self._views = _CartesianViews(xyz, x, y, z)


    @property
    def xyz(self) -> Quantity:
        return self._views.xyz


    @property
    def x(self) -> Quantity:
        return self._views.x


    @property
    def y(self) -> Quantity:
        return self._views.y


    @property
    def z(self) -> Quantity:
        return self._views.z


class CartesianCoordinates(CartesianCoordinatesArray):
    @classmethod
    def new(cls, xyz: Union[Quantity, ndarray, Sequence],
                 type_: int=0,
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None)                               \
                 -> CartesianCoordinates:

        xyz = cls._as_quantity(xyz, units)

        if xyz.size != 3:
            raise ValueError('invalid data size')
        self = cls(xyz.units, type_, frame)
        self._views.xyz.magnitude[:] = xyz.magnitude

        return self


    def __init__(self, units: Quantity,
                       type_: int=0,
                       frame: Frame=ECEF):

        super().__init__(units, 1, type_, frame)


    @property
    def xyz(self) -> Quantity:
        return self._views.xyz[0]


    @xyz.setter
    def xyz(self, value: Quantity) -> None:
        self._views.xyz[0,:] = value


    @property
    def x(self) -> Quantity:
        return self._views.x[0]


    @x.setter
    def x(self, value: Quantity) -> None:
        self._views.x[0] = value


    @property
    def y(self) -> Quantity:
        return self._views.y[0]


    @y.setter
    def y(self, value: Quantity) -> None:
        self._views.y[0] = value


    @property
    def z(self) -> Quantity:
        return self._views.z[0]


    @z.setter
    def z(self, value: Quantity) -> None:
        self._views.z[0] = value


class _SphericalViews(NamedTuple):
    r: Quantity
    theta: Quantity
    phi: Quantity


class SphericalCoordinatesArray(CoordinatesArray):
    _system = lib.GRAND_COORDINATES_SPHERICAL

    @classmethod
    def new(cls, r: Union[Quantity, ndarray, Sequence],
                 theta: Union[Quantity, ndarray, Sequence],
                 phi: Union[Quantity, ndarray, Sequence],
                 type_: int=0,
                 frame: Frame=ECEF,
                 units: Optional[Tuple[Quantity]]=None)                        \
                 -> SphericalCoordinatesArray:

        r, theta, phi = cls._as_quantity(r, theta, phi, units)

        self = cls((r.units, theta.units, phi.units), r.size, type_, frame)
        self._views.r.magnitude[:] = r.magnitude
        self._views.theta.magnitude[:] = theta.magnitude
        self._views.phi.magnitude[:] = phi.magnitude

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
                       type_: int=0, 
                       frame: Frame=ECEF):

        super().__init__(size, type_, self._system, frame)
        self._set_views(units)


    def _set_views(self, units: Tuple[Quantity]):
        r = Quantity(self._data[::3], units[0])
        theta = Quantity(self._data[1::3], units[1])
        phi = Quantity(self._data[2::3], units[2])
        self._views = _SphericalViews(r, theta, phi)


    @property
    def r(self) -> Quantity:
        return self._views.r


    @property
    def theta(self) -> Quantity:
        return self._views.theta


    @property
    def phi(self) -> Quantity:
        return self._views.phi


class SphericalCoordinates(SphericalCoordinatesArray):
    @classmethod
    def new(cls, r: Union[Quantity, float],
                 theta: Union[Quantity, float],
                 phi: Union[Quantity, float],
                 type_: int=0,
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

        self = cls((r.units, theta.units, phi.units), type_, frame)
        self._views.r.magnitude[0] = r.magnitude
        self._views.theta.magnitude[0] = theta.magnitude
        self._views.phi.magnitude[0] = phi.magnitude

        return self


    def __init__(self, units: Quantity,
                       type_: int=0,
                       frame: Frame=ECEF):

        super().__init__(units, 1, type_, frame)


    @property
    def r(self) -> Quantity:
        return self._views.r[0]


    @r.setter
    def r(self, value: Quantity) -> None:
        self._views.r[0] = value


    @property
    def theta(self) -> Quantity:
        return self._views.theta[0]


    @theta.setter
    def theta(self, value: Quantity) -> None:
        self._views.theta[0] = value


    @property
    def phi(self) -> Quantity:
        return self._views.phi[0]


    @phi.setter
    def phi(self, value: Quantity) -> None:
        self._views.phi[0] = value
