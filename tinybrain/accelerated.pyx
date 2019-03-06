"""
Cython accelerated routines for common downsampling operations.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: March 2019
"""

cimport cython
from cython cimport floating
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport numpy as np
import numpy as np

cdef extern from "accelerated.hpp" namespace "accelerated":
  cdef U* accumulate_2x2[T, U](T* arr, size_t sx, size_t sy, size_t sz)

def average_pooling_2x2(channel, uint32_t num_mips=1):
  while channel.ndim < 4:
    channel = channel[..., np.newaxis]

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]

  if min(sx, sy) <= 2 ** num_mips:
    raise ValueError("Can't downsample smaller than the smallest XY plane dimension.")

  if channel.dtype == np.uint8:
    return _average_pooling_2x2_uint8(channel, num_mips)
  if channel.dtype == np.uint16:
    return _average_pooling_2x2_uint16(channel, num_mips)
  elif channel.dtype == np.float32:
    return _average_pooling_2x2_float(channel, num_mips)
  elif channel.dtype == np.float64:
    return _average_pooling_2x2_double(channel, num_mips)
  else:
    raise TypeError("Unsupported data type: ", channel.dtype)

def render_image_uint8(uint16_t[:] accum, uint32_t bitshift, size_t ovoxels):
  cdef np.ndarray[uint8_t, ndim=1] oimg = np.zeros( (ovoxels,), dtype=np.uint8 )
  cdef size_t i = 0
  for i in range(ovoxels):
    oimg[i] = <uint8_t>(accum[i] >> bitshift)
  return oimg

def render_image_uint16(uint32_t[:] accum, uint32_t bitshift, size_t ovoxels):
  cdef np.ndarray[uint16_t, ndim=1] oimg = np.zeros( (ovoxels,), dtype=np.uint16 )
  cdef size_t i = 0
  for i in range(ovoxels):
    oimg[i] = <uint16_t>(accum[i] >> bitshift)
  return oimg

def render_image_flt32(float[:] accum, float divisor, size_t ovoxels):
  cdef np.ndarray[float, ndim=1] oimg = np.zeros( (ovoxels,), dtype=np.float32 )
  cdef size_t i = 0
  for i in range(ovoxels):
    oimg[i] = <float>(accum[i] / divisor)
  return oimg

def render_image_flt64(double[:] accum, double divisor, size_t ovoxels):
  cdef np.ndarray[double, ndim=1] oimg = np.zeros( (ovoxels,), dtype=np.float64 )
  cdef size_t i = 0
  for i in range(ovoxels):
    oimg[i] = <double>(accum[i] / divisor)
  return oimg

def _average_pooling_2x2_uint8(np.ndarray[uint8_t, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef uint8_t[:,:,:,:] channelview = channel
  cdef uint16_t* accum = accumulate_2x2[uint8_t, uint16_t](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef uint16_t[:] accumview = <uint16_t[:ovoxels]>accum
  cdef uint16_t* tmp
  cdef uint32_t mip, bitshift

  results = []
  for mip in range(num_mips):
    bitshift = 2 * ((mip % 4) + 1) # integer truncation every 4 mip levels
    oimg = render_image_uint8(accumview, bitshift, ovoxels)
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
    accum = accumulate_2x2[uint16_t, uint16_t](accum, sx, sy, sz, sw)
    accumview = <uint16_t[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

def _average_pooling_2x2_uint16(np.ndarray[uint16_t, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef uint16_t[:,:,:,:] channelview = channel
  cdef uint32_t* accum = accumulate_2x2[uint16_t, uint32_t](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef uint32_t[:] accumview = <uint32_t[:ovoxels]>accum
  cdef uint32_t* tmp
  cdef uint32_t mip, bitshift

  results = []
  for mip in range(num_mips):
    bitshift = 2 * ((mip % 4) + 1) # integer truncation every 4 mip levels
    oimg = render_image_int(accumview, bitshift, ovoxels)
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
    accum = accumulate_2x2[uint32_t, uint32_t](accum, sx, sy, sz, sw)
    accumview = <uint32_t[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

def _average_pooling_2x2_float(np.ndarray[float, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef float[:,:,:,:] channelview = channel
  cdef float* accum = accumulate_2x2[float, float](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef float[:] accumview = <float[:ovoxels]>accum
  cdef float* tmp
  cdef uint32_t mip

  cdef float divisor = 1.0

  results = []
  for mip in range(num_mips):
    divisor = 4.0 ** (mip+1)
    oimg = render_image_flt32(accumview, divisor, ovoxels)
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

    tmp = accum 
    accum = accumulate_2x2[float, float](accum, sx, sy, sz, sw)
    accumview = <float[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

def _average_pooling_2x2_double(np.ndarray[double, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef double[:,:,:,:] channelview = channel
  cdef double* accum = accumulate_2x2[double, double](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef double[:] accumview = <double[:ovoxels]>accum
  cdef double* tmp
  cdef uint32_t mip

  cdef double divisor = 1.0

  results = []
  for mip in range(num_mips):
    divisor = 4.0 ** (mip+1)
    oimg = render_image_flt64(accumview, divisor, ovoxels)
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

    tmp = accum 
    accum = accumulate_2x2[double, double](accum, sx, sy, sz, sw)
    accumview = <double[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results






