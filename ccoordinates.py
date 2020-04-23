from __future__ import annotations

from typing import Optional, Sequence, Union
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


ECEF = Frame()


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
        return int(self._c.size)


class CartesianCoordinatesArray(CoordinatesArray):
    @classmethod
    def new(cls, xyz: Union[ndarray, Sequence],
                 frame: Frame=ECEF,
                 units: Optional[Quantity]=None):

        if units is not None:
            xyz = Quantity(xyz, units)
        elif not isinstance(xyz, Quantity):
            raise TypeError('missing unit')

        size = xyz.size // 3 # XXX check size
        self = cls(size, frame, xyz.units)
        self.xyz.magnitude[:] = xyz.magnitude

        return self


    def __init__(self, size: int=0,
                       frame: Frame=ECEF,
                       units: Optional[Quantity]=None):

        super().__init__(size, lib.GRAND_COORDINATES_POINT,
                         lib.GRAND_COORDINATES_CARTESIAN, frame)

        self._xyz = Quantity(numpy.reshape(self._data, (size, 3)), units)
        self._x = self._xyz[:,0]
        self._y = self._xyz[:,1]
        self._z = self._xyz[:,2]


    @property
    def xyz(self) -> Quantity:
        return self._xyz


    @property
    def x(self) -> Quantity:
        return self._x


    @property
    def y(self) -> Quantity:
        return self._y


    @property
    def z(self) -> Quantity:
        return self._z


class CartesianCoordinates(CartesianCoordinatesArray):
    pass # XXX Use as a scalar
