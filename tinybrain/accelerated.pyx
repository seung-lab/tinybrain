"""
Cython accelerated routines for common downsampling operations.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: March 2019
"""
cimport cython
from cython cimport floating
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)
import ctypes

from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport numpy as np
import numpy as np
np.import_array()

cdef extern from "accelerated.hpp" namespace "accelerated":
  cdef T* _average_pooling_2x2_single_mip[T](T* arr, size_t sx, size_t sy, size_t sz, size_t sw, T* out)
  cdef U* accumulate_2x2[T, U](T* arr, size_t sx, size_t sy, size_t sz, size_t sw)
  cdef T* accumulate_2x2f[T](T* arr, size_t sx, size_t sy, size_t sz, size_t sw)
  cdef T* _average_pooling_2x2x2_single_mip[T](T* arr, size_t sx, size_t sy, size_t sz, size_t sw, T* out)
  cdef U* accumulate_2x2x2[T, U](T* arr, size_t sx, size_t sy, size_t sz, size_t sw)
  cdef U* denominator_2x2x2[T, U](T* arr, size_t sx, size_t sy, size_t sz, size_t sw)
  cdef void render_image[T, U](T* arr, U* oimg, uint32_t bitshift, size_t ovoxels)
  cdef void render_image_sparse[T, U](T* numerator, T* denominator, U* oimg, size_t ovoxels)
  cdef void render_image_floating[T](T* arr, T* oimg, T divisor, size_t ovoxels)
  cdef T* shift_right[T](T* arr, size_t ovoxels, size_t bits)
  cdef void _mode_pooling_2x2x2[T](
    T* img, T* oimg, 
    size_t sx, size_t sy, 
    size_t sz, size_t sw,
    uint8_t sparse
  )

def expand_dims(img, ndim):
  while img.ndim < ndim:
    img = img[..., np.newaxis]
  return img

def squeeze_dims(img, ndim):
  while img.ndim > ndim:
    img = img[..., 0]
  return img

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

ctypedef fused INTEGER:
  UINT
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused NUMBER:
  INTEGER
  float
  double

### AVERAGE POOLING 2x2 ####

def average_pooling_2x2(channel, size_t num_mips=1):
  ndim = channel.ndim
  channel = expand_dims(channel, 4)

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]

  if min(sx, sy) < <size_t>(2 ** num_mips):
    raise ValueError("Can't downsample smaller than the smallest XY plane dimension.")

  results = []
  if num_mips == 1:
    results = _average_pooling_2x2_single_mip_py(channel)
  elif channel.dtype == np.uint8:
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

def _average_pooling_2x2_single_mip_py(np.ndarray[NUMBER, ndim=4] channel):
  cdef uint8_t[:,:,:,:] arr_memview8u
  cdef uint16_t[:,:,:,:] arr_memview16u
  cdef uint32_t[:,:,:,:] arr_memview32u
  cdef uint64_t[:,:,:,:] arr_memview64u
  cdef float[:,:,:,:] arr_memviewf
  cdef double[:,:,:,:] arr_memviewd

  cdef uint8_t[:,:,:,:] out_memview8u
  cdef uint16_t[:,:,:,:] out_memview16u
  cdef uint32_t[:,:,:,:] out_memview32u
  cdef uint64_t[:,:,:,:] out_memview64u
  cdef float[:,:,:,:] out_memviewf
  cdef double[:,:,:,:] out_memviewd

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]

  shape = ( (sx+1)//2, (sy+1)//2, sz, sw )

  cdef np.ndarray[NUMBER, ndim=4] out = np.zeros(shape, dtype=channel.dtype, order="F")

  if channel.dtype == np.uint8:
    arr_memview8u = channel
    out_memview8u = out
    _average_pooling_2x2_single_mip[uint8_t](&arr_memview8u[0,0,0,0], sx, sy, sz, sw, &out_memview8u[0,0,0,0])
  elif channel.dtype == np.uint16:
    arr_memview16u = channel
    out_memview16u = out
    _average_pooling_2x2_single_mip[uint16_t](&arr_memview16u[0,0,0,0], sx, sy, sz, sw, &out_memview16u[0,0,0,0])
  elif channel.dtype == np.uint32:
    arr_memview32u = channel
    out_memview32u = out
    _average_pooling_2x2_single_mip[uint32_t](&arr_memview32u[0,0,0,0], sx, sy, sz, sw, &out_memview32u[0,0,0,0])
  elif channel.dtype == np.uint64:
    arr_memview64u = channel
    out_memview64u = out
    _average_pooling_2x2_single_mip[uint64_t](&arr_memview64u[0,0,0,0], sx, sy, sz, sw, &out_memview64u[0,0,0,0])
  elif channel.dtype == np.float32:
    arr_memviewf = channel
    out_memviewf = out
    _average_pooling_2x2_single_mip[float](&arr_memviewf[0,0,0,0], sx, sy, sz, sw, &out_memviewf[0,0,0,0])
  elif channel.dtype == np.float64:
    arr_memviewd = channel
    out_memviewd = out
    _average_pooling_2x2_single_mip[double](&arr_memviewd[0,0,0,0], sx, sy, sz, sw, &out_memviewd[0,0,0,0])
  else:
    raise TypeError("Unsupported data type. " + str(channel.dtype))

  return [ out ]

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

  cdef uint8_t[:] oimgview

  results = []
  for mip in range(num_mips):
    bitshift = 2 * ((mip % 4) + 1) # integer truncation every 4 mip levels
    oimg = np.zeros( (ovoxels,), dtype=np.uint8, order='F')
    oimgview = oimg
    render_image[uint16_t, uint8_t](&accumview[0], &oimgview[0], bitshift, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, sz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    if bitshift == 8:
      shift_right[uint16_t](accum, ovoxels, bitshift)

    sx = osx 
    sy = osy 
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * sz * sw

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

  cdef uint16_t[:] oimgview

  results = []
  for mip in range(num_mips):
    bitshift = 2 * ((mip % 4) + 1) # integer truncation every 4 mip levels
    oimg = np.zeros( (ovoxels,), dtype=np.uint16, order='F')
    oimgview = oimg
    render_image[uint32_t, uint16_t](&accumview[0], &oimgview[0], bitshift, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, sz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    if bitshift == 8:
      shift_right[uint32_t](accum, ovoxels, bitshift)

    sx = osx 
    sy = osy 
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * sz * sw

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
  cdef float* accum = accumulate_2x2f[float](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef float[:] accumview = <float[:ovoxels]>accum
  cdef float* tmp
  cdef uint32_t mip

  cdef float divisor = 1.0
  cdef float[:] oimgview

  results = []
  for mip in range(num_mips):
    divisor = 4.0 ** (mip+1)
    oimg = np.zeros( (ovoxels,), dtype=np.float32, order='F')
    oimgview = oimg
    render_image_floating[float](&accumview[0], &oimgview[0], divisor, ovoxels)

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
    accum = accumulate_2x2f[float](accum, sx, sy, sz, sw)
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
  cdef double* accum = accumulate_2x2f[double](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef double[:] accumview = <double[:ovoxels]>accum
  cdef double* tmp
  cdef uint32_t mip

  cdef double divisor = 1.0
  cdef double[:] oimgview

  results = []
  for mip in range(num_mips):
    divisor = 4.0 ** (mip+1)
    oimg = np.zeros( (ovoxels,), dtype=np.float64, order='F')
    oimgview = oimg
    render_image_floating[double](&accumview[0], &oimgview[0], divisor, ovoxels)

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
    accum = accumulate_2x2f[double](accum, sx, sy, sz, sw)
    accumview = <double[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

### AVERAGE POOLING 2x2x2 ###

def average_pooling_2x2x2(channel, size_t num_mips=1, sparse=False):
  ndim = channel.ndim
  channel = expand_dims(channel, 4)

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]

  if min(sx, sy, sz) < <size_t>(2 ** num_mips):
    raise ValueError("Can't downsample using a patch larger than the smallest plane dimension: <{},{},{}> {}".format(sx,sy,sz, 2**num_mips))

  if sparse and channel.dtype not in (np.uint8, np.uint16):
    raise ValueError("Sparse mode is only supported for uint8 and uint16 inputs. Got: " + str(channel.dtype))

  results = []
  if num_mips == 1 and not sparse:
    results = _average_pooling_2x2x2_single_mip_py(channel)
  elif channel.dtype == np.uint8:
    results = _average_pooling_2x2x2_uint8(channel, num_mips, sparse)
  elif channel.dtype == np.uint16:
    results = _average_pooling_2x2x2_uint16(channel, num_mips, sparse)
  elif channel.dtype == np.float32:
    results = _average_pooling_2x2x2_float(channel, num_mips)
  elif channel.dtype == np.float64:
    results = _average_pooling_2x2x2_double(channel, num_mips)
  else:
    raise TypeError("Unsupported data type: ", channel.dtype)

  for i, img in enumerate(results):
    results[i] = squeeze_dims(img, ndim)

  return results

def _average_pooling_2x2x2_single_mip_py(np.ndarray[NUMBER, ndim=4] channel):
  cdef uint8_t[:,:,:,:] arr_memview8u
  cdef uint16_t[:,:,:,:] arr_memview16u
  cdef uint32_t[:,:,:,:] arr_memview32u
  cdef uint64_t[:,:,:,:] arr_memview64u
  cdef float[:,:,:,:] arr_memviewf
  cdef double[:,:,:,:] arr_memviewd

  cdef uint8_t[:,:,:,:] out_memview8u
  cdef uint16_t[:,:,:,:] out_memview16u
  cdef uint32_t[:,:,:,:] out_memview32u
  cdef uint64_t[:,:,:,:] out_memview64u
  cdef float[:,:,:,:] out_memviewf
  cdef double[:,:,:,:] out_memviewd

  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]

  shape = ( (sx+1)//2, (sy+1)//2, (sz+1)//2, sw )

  cdef np.ndarray[NUMBER, ndim=4] out = np.zeros(shape, dtype=channel.dtype, order="F")

  if channel.dtype == np.uint8:
    arr_memview8u = channel
    out_memview8u = out
    _average_pooling_2x2x2_single_mip[uint8_t](&arr_memview8u[0,0,0,0], sx, sy, sz, sw, &out_memview8u[0,0,0,0])
  elif channel.dtype == np.uint16:
    arr_memview16u = channel
    out_memview16u = out
    _average_pooling_2x2x2_single_mip[uint16_t](&arr_memview16u[0,0,0,0], sx, sy, sz, sw, &out_memview16u[0,0,0,0])
  elif channel.dtype == np.uint32:
    arr_memview32u = channel
    out_memview32u = out
    _average_pooling_2x2x2_single_mip[uint32_t](&arr_memview32u[0,0,0,0], sx, sy, sz, sw, &out_memview32u[0,0,0,0])
  elif channel.dtype == np.uint64:
    arr_memview64u = channel
    out_memview64u = out
    _average_pooling_2x2x2_single_mip[uint64_t](&arr_memview64u[0,0,0,0], sx, sy, sz, sw, &out_memview64u[0,0,0,0])
  elif channel.dtype == np.float32:
    arr_memviewf = channel
    out_memviewf = out
    _average_pooling_2x2x2_single_mip[float](&arr_memviewf[0,0,0,0], sx, sy, sz, sw, &out_memviewf[0,0,0,0])
  elif channel.dtype == np.float64:
    arr_memviewd = channel
    out_memviewd = out
    _average_pooling_2x2x2_single_mip[double](&arr_memviewd[0,0,0,0], sx, sy, sz, sw, &out_memviewd[0,0,0,0])
  else:
    raise TypeError("Unsupported data type. " + str(channel.dtype))

  return [ out ]

def _average_pooling_2x2x2_uint8(np.ndarray[uint8_t, ndim=4] channel, uint32_t num_mips, sparse=False):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osz = (sz + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * osz * sw

  cdef uint8_t[:,:,:,:] channelview = channel
  cdef uint32_t* accum = accumulate_2x2x2[uint8_t, uint32_t](
    &channelview[0,0,0,0], sx, sy, sz, sw
  )
  cdef uint32_t[:] accumview = <uint32_t[:ovoxels]>accum

  # "denominator"
  cdef uint32_t* denom
  cdef uint32_t[:] denomview
  if sparse:
    denom = denominator_2x2x2[uint8_t, uint32_t](
      &channelview[0,0,0,0], sx, sy, sz, sw
    )
    denomview = <uint32_t[:ovoxels]>denom

  cdef uint32_t* tmp
  cdef uint32_t mip, bitshift

  cdef uint8_t[:] oimgview

  results = []
  for mip in range(num_mips):
    bitshift = 3 * ((mip % 8) + 1) # integer truncation every 8 mip levels
    oimg = np.zeros( (ovoxels,), dtype=np.uint8, order='F')
    oimgview = oimg

    if sparse:
      render_image_sparse[uint32_t, uint8_t](&accumview[0], &denomview[0], &oimgview[0], ovoxels)
    else:
      render_image[uint32_t, uint8_t](&accumview[0], &oimgview[0], bitshift, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, osz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    if bitshift == 24:
      shift_right[uint32_t](accum, ovoxels, bitshift)
      if sparse:
        shift_right[uint32_t](denom, ovoxels, bitshift)

    sx = osx 
    sy = osy 
    sz = osz
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osz = (sz + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * osz * sw

    tmp = accum 
    accum = accumulate_2x2x2[uint32_t, uint32_t](accum, sx, sy, sz, sw)
    accumview = <uint32_t[:ovoxels]>accum
    PyMem_Free(tmp)

    if sparse:
      tmp = denom
      denom = accumulate_2x2x2[uint32_t, uint32_t](denom, sx, sy, sz, sw)
      denomview = <uint32_t[:ovoxels]>denom
      PyMem_Free(tmp)

  PyMem_Free(accum)
  if sparse:
    PyMem_Free(denom)

  return results

def _average_pooling_2x2x2_uint16(np.ndarray[uint16_t, ndim=4] channel, uint32_t num_mips, sparse):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osz = (sz + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * osz * sw

  cdef uint16_t[:,:,:,:] channelview = channel
  cdef uint32_t* accum = accumulate_2x2x2[uint16_t, uint32_t](
    &channelview[0,0,0,0], sx, sy, sz, sw
  )
  cdef uint32_t[:] accumview = <uint32_t[:ovoxels]>accum

  # "denominator"
  cdef uint32_t* denom
  cdef uint32_t[:] denomview
  if sparse:
    denom = denominator_2x2x2[uint16_t, uint32_t](
      &channelview[0,0,0,0], sx, sy, sz, sw
    )
    denomview = <uint32_t[:ovoxels]>denom

  cdef uint32_t* tmp
  cdef uint32_t mip, bitshift

  cdef uint16_t[:] oimgview

  results = []
  for mip in range(num_mips):
    bitshift = 3 * ((mip % 5) + 1) # integer truncation every 5 mip levels
    oimg = np.zeros( (ovoxels,), dtype=np.uint16, order='F')
    oimgview = oimg
    if sparse:
      render_image_sparse[uint32_t, uint16_t](&accumview[0], &denomview[0], &oimgview[0], ovoxels)
    else:
      render_image[uint32_t, uint16_t](&accumview[0], &oimgview[0], bitshift, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, osz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    if bitshift == 15:
      shift_right[uint32_t](accum, ovoxels, bitshift)
      if sparse:
        shift_right[uint32_t](denom, ovoxels, bitshift)

    sx = osx 
    sy = osy 
    sz = osz
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osz = (sz + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * osz * sw

    tmp = accum 
    accum = accumulate_2x2x2[uint32_t, uint32_t](accum, sx, sy, sz, sw)
    accumview = <uint32_t[:ovoxels]>accum
    PyMem_Free(tmp)

    if sparse:
      tmp = denom
      denom = accumulate_2x2x2[uint32_t, uint32_t](denom, sx, sy, sz, sw)
      denomview = <uint32_t[:ovoxels]>denom
      PyMem_Free(tmp)

  PyMem_Free(accum)
  if sparse:
    PyMem_Free(denom)

  return results

def _average_pooling_2x2x2_float(np.ndarray[float, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osz = (sz + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * osz * sw
  
  cdef float[:,:,:,:] channelview = channel
  cdef float* accum = accumulate_2x2x2[float,float](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef float[:] accumview = <float[:ovoxels]>accum
  cdef float* tmp
  cdef uint32_t mip

  cdef float divisor = 1.0
  cdef float[:] oimgview

  results = []
  for mip in range(num_mips):
    divisor = 8.0 ** (mip+1)
    oimg = np.zeros( (ovoxels,), dtype=channel.dtype, order='F')
    oimgview = oimg
    render_image_floating[float](&accumview[0], &oimgview[0], divisor, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, osz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    sx = osx 
    sy = osy 
    sz = osz
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osz = (sz + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * osz * sw

    tmp = accum 
    accum = accumulate_2x2x2[float,float](accum, sx, sy, sz, sw)
    accumview = <float[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

def _average_pooling_2x2x2_double(np.ndarray[double, ndim=4] channel, uint32_t num_mips):
  cdef size_t sx = channel.shape[0]
  cdef size_t sy = channel.shape[1]
  cdef size_t sz = channel.shape[2]
  cdef size_t sw = channel.shape[3]
  cdef size_t sxy = sx * sy

  cdef size_t osx = (sx + 1) // 2
  cdef size_t osy = (sy + 1) // 2
  cdef size_t osz = (sz + 1) // 2
  cdef size_t osxy = osx * osy
  cdef size_t ovoxels = osxy * osz * sw
  
  cdef double[:,:,:,:] channelview = channel
  cdef double* accum = accumulate_2x2x2[double,double](&channelview[0,0,0,0], sx, sy, sz, sw)
  cdef double[:] accumview = <double[:ovoxels]>accum
  cdef double* tmp
  cdef uint32_t mip

  cdef double divisor = 1.0
  cdef double[:] oimgview

  results = []
  for mip in range(num_mips):
    divisor = 8.0 ** (mip+1)
    oimg = np.zeros( (ovoxels,), dtype=channel.dtype, order='F')
    oimgview = oimg
    render_image_floating[double](&accumview[0], &oimgview[0], divisor, ovoxels)

    results.append(
      oimg.reshape( (osx, osy, osz, sw), order='F' )
    )

    if mip == num_mips - 1:
      break

    sx = osx 
    sy = osy 
    sz = osz
    sxy = sx * sy
    osx = (sx + 1) // 2
    osy = (sy + 1) // 2
    osz = (sz + 1) // 2
    osxy = osx * osy
    ovoxels = osxy * osz * sw

    tmp = accum 
    accum = accumulate_2x2x2[double,double](accum, sx, sy, sz, sw)
    accumview = <double[:ovoxels]>accum
    PyMem_Free(tmp)

  PyMem_Free(accum)

  return results

### MODE POOLING ####

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

def mode_pooling_2x2x2(img, uint32_t num_mips=1, sparse=False):
  ndim = img.ndim
  img = expand_dims(img, 4)

  results = []
  for mip in range(num_mips):
    img = _mode_pooling_2x2x2_helper(img, sparse)
    results.append(img)

  for i, img in enumerate(results):
    results[i] = squeeze_dims(img, ndim)

  return results  

def _mode_pooling_2x2x2_helper(np.ndarray[NUMBER, ndim=4] img, sparse):
  sx = img.shape[0]
  sy = img.shape[1]
  sz = img.shape[2]
  sw = img.shape[3]

  size = ( (sx+1) // 2, (sy+1) // 2, (sz+1) // 2, sw )

  cdef np.ndarray[NUMBER, ndim=4] oimg = np.zeros(size, dtype=img.dtype, order='F')

  cdef uint8_t[:,:,:,:] arr_memview8u_i
  cdef uint16_t[:,:,:,:] arr_memview16u_i
  cdef uint32_t[:,:,:,:] arr_memview32u_i
  cdef uint64_t[:,:,:,:] arr_memview64u_i

  cdef uint8_t[:,:,:,:] arr_memview8u_o
  cdef uint16_t[:,:,:,:] arr_memview16u_o
  cdef uint32_t[:,:,:,:] arr_memview32u_o
  cdef uint64_t[:,:,:,:] arr_memview64u_o

  if img.dtype in (np.uint8, np.int8):
    arr_memview8u_i = img.view(np.uint8)
    arr_memview8u_o = oimg.view(np.uint8)
    _mode_pooling_2x2x2[uint8_t](
      &arr_memview8u_i[0,0,0,0], &arr_memview8u_o[0,0,0,0], 
      sx, sy, sz, sw, 
      bool(sparse)
    )
  elif img.dtype in (np.uint16, np.int16):
    arr_memview16u_i = img.view(np.uint16)
    arr_memview16u_o = oimg.view(np.uint16)
    _mode_pooling_2x2x2[uint16_t](
      &arr_memview16u_i[0,0,0,0], &arr_memview16u_o[0,0,0,0], 
      sx, sy, sz, sw,
      bool(sparse)
    )
  elif img.dtype in (np.uint32, np.int32, np.float32):
    arr_memview32u_i = img.view(np.uint32)
    arr_memview32u_o = oimg.view(np.uint32)
    _mode_pooling_2x2x2[uint32_t](
      &arr_memview32u_i[0,0,0,0], &arr_memview32u_o[0,0,0,0], 
      sx, sy, sz, sw,
      bool(sparse)
    )
  elif img.dtype in (np.uint64, np.int64, np.float64, np.csingle):
    arr_memview64u_i = img.view(np.uint64)
    arr_memview64u_o = oimg.view(np.uint64)
    _mode_pooling_2x2x2[uint64_t](
      &arr_memview64u_i[0,0,0,0], &arr_memview64u_o[0,0,0,0], 
      sx, sy, sz, sw,
      bool(sparse)
    )
  else:
    raise ValueError("{} not supported.".format(img.dtype))

  return oimg






