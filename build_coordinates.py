from cffi import FFI
import os
from pathlib import Path
from pycparser import parse_file, c_generator
import sys


PREFIX = Path(__file__).parent.resolve()
SRC_DIR = PREFIX / 'src'

BUILD_DIR = Path('build').resolve()
TMP_DIR = BUILD_DIR / 'tmp'


ffi = FFI()


def include(path, **opts):
    args = []
    for k, v in opts.items():
        args.append(f'-D{k}={v}')
    args.append('-I' + str(SRC_DIR))
    args.append('-DFILE=struct FILE')
    args.append('-O0')
    args.append('-g')

    ast = parse_file(str(path), use_cpp = True, cpp_args = args)
    generator = c_generator.CGenerator()
    header = generator.visit(ast)
    ffi.cdef(header)

include(SRC_DIR / 'coordinates.h')


def configure():
    with open(SRC_DIR / 'coordinates.c') as f:
        ffi.set_source('_coordinates',
            f.read(),
            include_dirs = [str(SRC_DIR)]
        )

configure()


def build():
    TMP_DIR.mkdir(parents = True, exist_ok = True)

    os.chdir(TMP_DIR)
    module = Path(ffi.compile(verbose=False))
    module = module.rename(PREFIX / '_coordinates.abi3.so')


if __name__ == '__main__':
    build()

