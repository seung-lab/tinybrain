import setuptools
import sys 
import platform

# NOTE: If accelerated.cpp does not exist:
# cython -3 --fast-fail -v --cplus tinybrain/accelerated.pyx

import numpy as np

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
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  python_requires="~=3.6",
  ext_modules=[
    setuptools.Extension(
      'tinybrain.accelerated',
      sources=[ 'tinybrain/accelerated.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=extra_compile_args,
    )
  ],
  long_description_content_type='text/markdown',
  pbr=True)





