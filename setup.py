import setuptools
import sys 
import platform

# NOTE: If accelerated.cpp does not exist:
# cython -3 --fast-fail -v --cplus tinybrain/accelerated.pyx

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__


extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
  if platform.machine() == "x86_64":
    extra_compile_args += [ "/arch:SSE3" ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3' # '-DCYTHON_TRACE=1'
  ]
  if platform.machine() == "x86_64":
    extra_compile_args += [ '-msse3' ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy','cython'],
  install_requires=['numpy'],
  python_requires=">=3.7",
  ext_modules=[
    setuptools.Extension(
      'tinybrain.accelerated',
      sources=[ 'tinybrain/accelerated.pyx' ],
      language='c++',
      include_dirs=[ str(NumpyImport()) ],
      extra_compile_args=extra_compile_args,
    )
  ],
  long_description_content_type='text/markdown',
  pbr=True)





