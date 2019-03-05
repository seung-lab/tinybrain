"""
Cython accelerated routines for common downsampling operations.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: March 2019
"""

cimport cython
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport numpy as np
import numpy as np

cdef extern from "accelerated.hpp" namespace "accelerated":
  cdef uint16_t* accumulate_2x2[T](T* arr, size_t sx, size_t sy, size_t sz)

def render_image(uint16_t[:] accum, uint32_t bitshift, size_t ovoxels):
  cdef np.ndarray[uint8_t, ndim=1] oimg = np.zeros( (ovoxels,), dtype=np.uint8 )
  cdef size_t i = 0
  for i in range(ovoxels):
    oimg[i] = <uint8_t>(accum[i] >> bitshift)
  return oimg

def average_pooling_2x2(
    np.ndarray[uint8_t, ndim=4] channel, 
    uint32_t num_mips=1
  ):

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  if num_mips == 0:
    return []

  if min(sx, sy) <= 2 ** num_mips:
    raise ValueError("Can't downsample smaller than the smallest XY plane dimension.")

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef uint8_t[:,:,:,:] channelview = channel
  cdef uint16_t* accum = accumulate_2x2[uint8_t](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef uint16_t[:] accumview = <uint16_t[:ovoxels]>accum
  cdef uint16_t* tmp
  cdef size_t i
  cdef uint32_t mip, bitshift

  results = []
  for mip in range(0, num_mips):
    bitshift = 2 * ((mip % 4) + 1) # integer truncation every 4 mip levels

    oimg = render_image(accumview, bitshift, ovoxels)
    results.append(
      oimg.reshape( (osx, osy, sz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    sx = osx 
    sy = osy 
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * sz * sw

    if bitshift == 8:
      for i in range(sx * sy * sz * sw):
        accum[i] >>= 8

    tmp = accum 
    accum = accumulate_2x2[uint16_t](accum, sx, sy, sz, sw)
    accumview = <uint16_t[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results










