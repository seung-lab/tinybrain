import setuptools

# NOTE: If accelerated.cpp does not exist:
# cython -3 --fast-fail -v --cplus tinybrain/accelerated.pyx

import numpy as np

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
      extra_compile_args=[
        '-std=c++11', '-O3', '-ffast-math', '-msse3',
        # '-DCYTHON_TRACE=1'
      ]
    )
  ],
  long_description_content_type='text/markdown',
  pbr=True)





