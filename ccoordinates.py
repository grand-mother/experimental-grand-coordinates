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
                 units: Optional[Quantity]=None):

        if units is not None:
            xyz = Quantity(xyz, units)
        elif not isinstance(xyz, Quantity):
            raise TypeError('missing unit')

        size = xyz.size // 3
        if 3 * size != xyz.size:
            raise ValueError('invalid data size')
        self = cls(size, type_, frame, xyz.units)
        self.xyz.magnitude[:] = xyz.magnitude

        return self


    def __init__(self, size: int=0,
                       type_: int=0, 
                       frame: Frame=ECEF,
                       units: Optional[Quantity]=None):

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
    def new(cls, xyz: Union[ndarray, Sequence],
                 type_: int=0,
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None):
        if units is not None:
            xyz = Quantity(xyz, units)
        elif not isinstance(xyz, Quantity):
            raise TypeError('missing unit')

        if xyz.size != 3:
            raise ValueError('invalid data size')
        self = cls(type_, frame, xyz.units)
        self.xyz.magnitude[:] = xyz.magnitude

        return self


    def __init__(self, type_: int=0, 
                       frame: Frame=ECEF,
                       units: Optional[Quantity]=None):

        super().__init__(1, type_, frame, units)


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
