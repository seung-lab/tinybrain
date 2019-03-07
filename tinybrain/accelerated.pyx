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

def expand_dims(img, ndim):
  while img.ndim < ndim:
    img = img[..., np.newaxis]
  return img

def squeeze_dims(img, ndim):
  while img.ndim > ndim:
    img = img[..., 0]
  return img

### AVERAGE POOLING ####

def average_pooling_2x2(channel, uint32_t num_mips=1):
  ndim = channel.ndim
  channel = expand_dims(channel, 4)

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]

  if min(sx, sy) <= 2 ** num_mips:
    raise ValueError("Can't downsample smaller than the smallest XY plane dimension.")

  results = []
  if channel.dtype == np.uint8:
    results = _average_pooling_2x2_uint8(channel, num_mips)
  elif channel.dtype == np.uint16:
    results = _average_pooling_2x2_uint16(channel, num_mips)
  elif channel.dtype == np.float32:
    results = _average_pooling_2x2_float(channel, num_mips)
  elif channel.dtype == np.float64:
    results = _average_pooling_2x2_double(channel, num_mips)
  else:
    raise TypeError("Unsupported data type: ", channel.dtype)

  for i, img in enumerate(results):
    results[i] = squeeze_dims(img, ndim)

  return results

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
    oimg = render_image_uint16(accumview, bitshift, ovoxels)
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

### MODE POOLING ####

ctypedef fused NUMBER:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t
  float
  double

def mode_pooling_2x2(img, uint32_t num_mips=1):
  ndim = img.ndim
  img = expand_dims(img, 4)

  results = []
  for mip in range(num_mips):
    img = _mode_pooling_2x2(img)
    results.append(img)

  for i, img in enumerate(results):
    results[i] = squeeze_dims(img, ndim)

  return results

def _mode_pooling_2x2(np.ndarray[NUMBER, ndim=4] img):
  cdef size_t sx = img.shape[0]
  cdef size_t sy = img.shape[1]
  cdef size_t sz = img.shape[2]
  cdef size_t sw = img.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * sz * sw

  cdef size_t x, y, z, w
  cdef NUMBER a, b, c, d 

  cdef np.ndarray[NUMBER, ndim=4] oimg = np.zeros( (osx, osy, sz, sw), dtype=img.dtype )

  cdef size_t ox, oy

  cdef size_t xodd = (sx & 0x01)
  cdef size_t yodd = (sy & 0x01)

  for w in range(sw):
    for z in range(sz):
      oy = 0
      y = 0
      for y in range(0, sy - yodd, 2):
        ox = 0
        for x in range(0, sx - xodd, 2):
          a = img[x  ,y  , z, w]
          b = img[x+1,y  , z, w]
          c = img[x  ,y+1, z, w]
          d = img[x+1,y+1, z, w]

          if a == b:
            oimg[ox, oy, z, w] = a
          elif a == c:
            oimg[ox, oy, z, w] = a
          elif b == c:
            oimg[ox, oy, z, w] = b
          else:
            oimg[ox, oy, z, w] = d

          ox += 1
        if xodd:
          oimg[ox, oy, z, w] = img[ sx - 1, y, z, w ]
          ox += 1
        oy += 1

      if yodd:
        for x in range(osx - xodd):
          oimg[x, oy, z, w] = img[ x*2, y, z, w ]
        if xodd:
          oimg[osx - 1, oy, z, w] = img[ sx - 1, y, z, w]

  return oimg.reshape( (osx, osy, sz, sw) )
