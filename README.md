# Experimental implementations of `grand.coordinates`

This repository contains experimental implementations of `grand.coordinates`.
It requires the `ctypes`, `grand` and `pint` packages and at least Python 3.7.
There are two implementations:

- [`coordinates.py`](coordinates.py) uses `grand.libs` for coordinates
  transforms (`libturtle`).

- [`ccoordinates.py`](ccoordinates.py) is an interface to what could be a C
  implementation of `grand_coordinates`. The corresponding C source is in the
  [src/](src) directory. The interfacing to C is done with the `cffi` package
  and data are exposed as `numpy.ndarray`.
  > Note that there are no coordinates transforms as for now. This
  > implementation simply illustrates how the same C data can be exposed as
  > different views using numpy, e.g. as a single `xyz` array or as separate
  > `x`, `y` and `z` arrays. This approach would also provide the same
  > coordinates objects at the C level with seamless interoperability with
  > Python.

In both cases `astropy` is not used. The `pint` package is a perfect
replacement for `astropy.units` and Earth coordinates transforms are available
from the C `TURTLE` library also used in the neutrino simulation. It was
cross-checked to be consistent with the `astropy` `AltAz`, `EarthLocation` and
`ITRS` coordinates.

# Usage

From the root of this repo and assuming that the `grand` repo is available
aside:
```bash
source ../grand/env/setup.sh
pip install ctypes pint
python build_coordinates.py

```
Then the [test.py](test.py) and [test-c.py](test-c.py) can be used in order
to test (showcase) the two implementations.
