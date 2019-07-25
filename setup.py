import setuptools
import sys 

# NOTE: If accelerated.cpp does not exist:
# cython -3 --fast-fail -v --cplus tinybrain/accelerated.pyx

import numpy as np


extra_compile_args = [
  '-std=c++11', '-O3', '-msse3',
  # '-DCYTHON_TRACE=1'
]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
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





