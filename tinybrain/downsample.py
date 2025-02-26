# The following license applies to functions: 
#     downsample_with_averaging_numpy
#     downsample_with_striding

# The license for the rest of the code is in GPLv3 in LICENSE

# @license
# Copyright 2017 The Neuroglancer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import combinations
from functools import reduce, partial

import math
import operator
import numpy as np

import tinybrain.accelerated

class DimensionError(Exception):
  pass

def downsample_with_averaging(img, factor, num_mips=1, sparse=False):
  """
  Downsample by average pooling patches of size factor. If sparse,
  omit background pixels from the denominator when averaging.

  Sparse mode is only supported for uint8 and uint16 type inputs.

  img: up to a 4D numpy image
  factor: e.g. (2,2,1) or (2,2,2)
  num_mips: generate this number of mip levels at once
  sparse: boolean, if True, don't consider background pixels

  Returns: [ mip0, mip1, mip2, ..., num_mip ]
  """
  if (
    img.dtype in (np.int8, np.int16, np.uint8, np.uint16, np.float32, np.float64)
    or num_mips == 1 # _average_pooling_2x2_single_mip_py supports all primative types
  ):
    img = np.asfortranarray(img)
    if (tuple(factor) in ( (2,2), (2,2,1), (2,2,1,1) )):
      return tinybrain.accelerated.average_pooling_2x2(img, num_mips, sparse)
    elif (tuple(factor) in ( (2,2,2), (2,2,2,1) )):
      return tinybrain.accelerated.average_pooling_2x2x2(img, num_mips, sparse)

  results = []
  if np.dtype(img.dtype).itemsize < 4:
    dtype = img.dtype
    img = img.astype(np.float32)
    for mip in range(num_mips):
      img = downsample_with_averaging_numpy(img, factor, sparse)
      results.append(img.astype(dtype))
  else:
    for mip in range(num_mips):
      img = downsample_with_averaging_numpy(img, factor, sparse)
      results.append(img)

  return results

def validate_factor(array, factor):
  factor = np.array(factor, dtype=np.int32)
  if np.any(factor <= 0):
    raise ValueError("Factors less than one don't make sense. Factor: {}".format(factor))

  factor = list(factor)
  while len(factor) < len(array.shape):
    factor += [ 1 ]

  return tuple(factor)

def odd_to_even2d(image):
  """
  To facilitate 2x2 downsampling segmentation, change an odd sized image into an even sized one.
  Works by mirroring the starting 1 pixel edge of the image on odd shaped sides.

  e.g. turn a 3x3x5 image into a 4x4x5 (the x and y are what are getting downsampled)
  
  For example: [ 3, 2, 4 ] => [ 3, 3, 2, 4 ] which is now super easy to downsample.

  """
  if len(image.shape) == 3:
    image = image[ :,:,:, np.newaxis ]

  shape = np.array(image.shape)

  offset = (shape % 2)[:2] # x,y offset
  
  if not np.any(offset): # any non-zeros
    return image

  oddshape = image.shape[:2] + offset
  oddshape = np.append(oddshape, shape[2:])
  oddshape = oddshape.astype(int)

  newimg = np.zeros(shape=oddshape, dtype=image.dtype)

  ox,oy = offset
  sx,sy,sz,ch = oddshape

  newimg[-1,-1,:] = image[-1,-1,:]
  newimg[:sx-ox,-1,:] = image[:,-1,:]
  newimg[-1,:sy-oy,:] = image[-1,:,:]
  newimg[:sx-ox,:sy-oy] = image
  
  return newimg

def downsample_with_averaging_numpy(array, factor, sparse=False):
  """
  Downsample x by factor using averaging.

  If factor has fewer parameters than data.shape, the remainder
  are assumed to be 1.

  @return: The downsampled array, of the same type as x.
  """
  factor = validate_factor(array, factor)
  if np.array_equal(factor[:3], np.array([1,1,1])):
    return array

  output_shape = tuple(int(math.ceil(s / f)) for s, f in zip(array.shape, factor))
  temp = np.zeros(output_shape, dtype=np.float32)
  counts = np.zeros(output_shape, int)
  for offset in np.ndindex(factor):
    part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    indexing_expr = tuple(np.s_[:s] for s in part.shape)
    temp[indexing_expr] += part
    if sparse:
      counts[indexing_expr] += part != 0
    else:
      counts[indexing_expr] += 1

  if sparse:
    np.maximum(counts, 1, out=counts)
  return np.asarray(temp / counts, dtype=array.dtype)

def downsample_with_max_pooling(array, factor, num_mips=1):
  """
  Downsample by picking the maximum value within a
  cuboid specified by factor. That is, a reduction factor
  of 2x2 works by summarizing many 2x2 cuboids. If factor's 
  length is smaller than array.shape, the remaining factors will
  be filled with 1.
  """
  results = []
  for mip in range(num_mips):
    array = _downsample_with(array, factor, fn=np.maximum)
    results.append(array)
  return results

def downsample_with_min_pooling(array, factor, num_mips=1):
  """
  Downsample by picking the minimum value within a
  cuboid specified by factor. That is, a reduction factor
  of 2x2 works by summarizing many 2x2 cuboids. If factor's 
  length is smaller than array.shape, the remaining factors will
  be filled with 1.
  """
  results = []
  for mip in range(num_mips):
    array = _downsample_with(array, factor, fn=np.minimum)
    results.append(array)
  return results

def _downsample_with(array, factor, fn):
  """
  Downsample by picking the maximum value within a
  cuboid specified by factor. That is, a reduction factor
  of 2x2 works by summarizing many 2x2 cuboids. If factor's 
  length is smaller than array.shape, the remaining factors will
  be filled with 1.
  """
  factor = validate_factor(array, factor)
  if np.all(np.array(factor, int) == 1):
      return array

  sections = []

  for offset in np.ndindex(factor):
    part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  output = sections[0].copy()

  for section in sections[1:]:
    fn(output, section, output)

  return output

def downsample_segmentation(img, factor, sparse=False, num_mips=1):
  """
  Downsampling machine labels requires choosing an actual
  pixel, not a linear combination (or otherwise) of the
  values contained within.

  Striding is the default choice for such situations, it is fast.
  However, a fairer way to do this is to pick the most frequent
  pixel in a patch. 

  Towards this end, the countless2d algorithm handles picking
  the most frequent pixel in 2^Nx2^Nx1 downsampling, while countless3d
  handles the 2^Nx2^Nx2^N case. Other factors (and non-integer datatypes)
  are handled by striding.

  countless2d & 3d both perform a 2x2 and 2x2x2 downsample, the higher power
  of two factors are handled by recursive calculation (so only the first
  downsample is fully accurate in the sense of being the mode).

  If factor has fewer parameters than img.shape, the remainder
  are assumed to be 1.

  Required:
    img: A 3d or 4d numpy array representing a segmentation image.
    factor: a tuple of downsample factors. Commonly: (2,2,1) to achieve
      a planar downsampling of 2x2 that preserves Z.

  Returns: a downsampled numpy array
  """
  factor = np.array(factor)
  if np.all(np.array(factor, int) == 1):
      return [ img ] * num_mips

  if tuple(factor) in ( (2,2), (2,2,1), (2,2,1,1) ) and not sparse:
    return tinybrain.accelerated.mode_pooling_2x2(img, num_mips=num_mips)
  elif tuple(factor) in ( (2,2,2), (2,2,2,1) ):
    return tinybrain.accelerated.mode_pooling_2x2x2(img, num_mips=num_mips, sparse=sparse)

  ndim = img.ndim
  img = expand_dims(img, 4)

  results = []
  for mip in range(num_mips):
    img = _downsample_segmentation(img, factor, sparse)
    results.append(img)

  results = [ squeeze_dims(img, ndim) for img in results ]
  return results

def _downsample_segmentation(data, factor, sparse=False):
  if np.all(np.array(factor) == 1):
    return data

  if data.dtype.kind not in ('u', 'i'): # integer types
    return downsample_with_striding(data, tuple(factor))[0]

  is_pot = lambda x: (x > 1) and not (np.log2(x) % 1) # is power of two
  is_twod_pot_downsample = np.any(factor == 1) and is_pot(reduce(operator.mul, factor))
  is_threed_pot_downsample = not np.any(factor == 1) and is_pot(reduce(operator.mul, factor)) 

  shape3d = np.array(data.shape[:3])
  modulo_shape = shape3d % 2
  # it's possible to write a 3d even to odd to make this 
  # work for all data shapes.
  if is_threed_pot_downsample and sum(modulo_shape) == 0: # power of two downsample on an even shape
    return _downsample_segmentation(countless3d(data), factor / 2)

  if not is_twod_pot_downsample:
    return downsample_with_striding(data, tuple(factor))[0]

  return downsample_segmentation_2d(data, factor, sparse)

def downsample_segmentation_2d(data, factor, sparse):
  """
  Call countless2d but manage the image to make it work
  for both even and odd sided images. Swap axes to enable
  alternate axis downsampling.
  """
  factor = np.array(factor, dtype=np.uint32)
  preserved_axis = np.where(factor == 1)[0][0] # e.g. 0, 1, 2

  shape3d = np.array(data.shape[:3])

  modulo_shape = shape3d % 2
  modulo_shape[preserved_axis] = 0
  has_even_dims = sum(modulo_shape) == 0 

  # algorithm is written for xy plane, so
  # switch other orientations to that plane, 
  # do computation and switch back.
  data = np.swapaxes(data, preserved_axis, 2)

  if not has_even_dims:
    data = odd_to_even2d(data)
    shape3d = np.array(data.shape[:3])

  output = np.zeros(
    shape=( int(data.shape[0] / 2), int(data.shape[1] / 2), data.shape[2], data.shape[3]), 
    dtype=data.dtype
  )

  if sparse:
    for z in range(data.shape[2]):
      output[:,:,z,:] = stippled_countless2d(data[:,:,z,:])
  else:
    for z in range(data.shape[2]):
      output[:,:,z,:] = countless2d(data[:,:,z,:])

  factor = np.array((factor / 2), dtype=np.int32)
  factor[preserved_axis] = 1

  output = np.swapaxes(output, preserved_axis, 2)
  
  return _downsample_segmentation(output, factor)

def countless2d(data):
  """
  Vectorized implementation of downsampling a 2D labeled
  image by 2 on each side using the COUNTLESS algorithm.

  countless2d is slightly faster and more memory efficent than the
  generalized algorithm countless.

  c.f. https://towardsdatascience.com/countless-high-performance-2x-downsampling-of-labeled-images-using-python-and-numpy-e70ad3275589
  """
  sections = []

  # allows us to prevent losing 1/2 a bit of information 
  # at the top end by using a bigger type. Without this 255 is handled incorrectly.
  data, upgraded = upgrade_type(data) 

  # offset from zero, raw countless doesn't handle 0 correctly
  # we'll remove the extra 1 at the end.
  data += 1

  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # ab := a if a == b else 0 and so on for ac, bc
  ab_ac |= b * (b == c)
  ab_ac += (ab_ac == 0) * d - 1 # a or d - 1

  if upgraded:
    return downgrade_type(ab_ac)

  # only need to reset data if we weren't upgraded 
  # b/c no copy was made
  data -= 1 

  return ab_ac

def stippled_countless2d(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm
  that treats zero as "background" and inflates lone
  pixels.
  
  https://medium.com/@willsilversmith/countless-2d-inflated-2x-downsampling-of-labeled-images-holding-zero-values-as-background-4d13a7675f2d
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  ab_ac |= b * (b == c) # PICK(B,C)
  nonzero = a + (a == 0) * (b + (b == 0) * c)
  return ab_ac + (ab_ac == 0) * (d + (d == 0) * nonzero) # AB || AC || BC || D

def countless3d(data):
  """return downsampled 2x2x2 data for even sided images."""
  ndim = data.ndim
  data = expand_dims(data, 4)
  modshape = np.array(data.shape[:3]) % 2
  if sum(modshape) != 0 or data.shape[3] != 1:
    raise DimensionError("COUNTLESS 3D currently only supports even sided images.") # someone has to write even_to_odd3d

  data = countless(data, (2,2,2))
  return squeeze_dims(data, ndim)

def countless(data, factor):
  """
  countless downsamples labeled images (segmentations)
  by finding the mode using vectorized instructions.

  It is ill advised to use this O(2^N-1) time algorithm
  and O(NCN/2) space for N > about 16 tops. 
  This means it's useful for the following kinds 
  of downsampling.

  This could be implemented for higher performance in
  C/Cython more simply, but at least this is easily
  portable.

  2x2x1 (N=4), 2x2x2 (N=8), 4x4x1 (N=16), 3x2x1 (N=6)
  and various other configurations of a similar nature.

  c.f. https://medium.com/@willsilversmith/countless-3d-vectorized-2x-downsampling-of-labeled-volume-images-using-python-and-numpy-59d686c2f75

  """
  sections = []

  mode_of = reduce(lambda x,y: x * y, factor)
  majority = int(math.ceil(float(mode_of) / 2))

  data += 1 # offset from zero
  
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  pick = lambda a,b: a * (a == b)
  lor = lambda x,y: x + (x == 0) * y # logical or

  subproblems = [ {}, {} ]
  results2 = None
  for x,y in combinations(range(len(sections) - 1), 2):
    res = pick(sections[x], sections[y])
    subproblems[0][(x,y)] = res
    if results2 is not None:
      results2 = lor(results2, res)
    else:
      results2 = res

  results = [ results2 ]
  for r in range(3, majority+1):
    r_results = None
    for combo in combinations(range(len(sections)), r):
      res = pick(subproblems[0][combo[:-1]], sections[combo[-1]])
      
      if combo[-1] != len(sections) - 1:
        subproblems[1][combo] = res

      if r_results is not None:
        r_results = lor(r_results, res)
      else:
        r_results = res
    results.append(r_results)
    subproblems[0] = subproblems[1]
    subproblems[1] = {}
    
  results.reverse()
  final_result = lor(reduce(lor, results), sections[-1]) - 1
  data -= 1
  return final_result


def upgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint8:
    return arr.astype(np.uint16), True
  elif dtype == np.uint16:
    return arr.astype(np.uint32), True
  elif dtype == np.uint32:
    return arr.astype(np.uint64), True

  return arr, False
  
def downgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint64:
    return arr.astype(np.uint32)
  elif dtype == np.uint32:
    return arr.astype(np.uint16)
  elif dtype == np.uint16:
    return arr.astype(np.uint8)
  
  return arr

def downsample_with_striding(array, factor, num_mips=1): 
  """
  Downsample x by factor using striding.

  If factor has fewer parameters than data.shape, the remainder
  are assumed to be 1.

  @return: The downsampled array, of the same type as x.
  """
  ndim = array.ndim 
  array = expand_dims(array, 4)

  factor = validate_factor(array, factor)
  if np.all(np.array(factor, int) == 1):
    return []

  results = []
  for mip in range(num_mips):
    array = array[tuple(np.s_[::f] for f in factor)]
    results.append( squeeze_dims(array, ndim) )
  
  return results 

def expand_dims(img, ndim):
  while img.ndim < ndim:
    img = img[..., np.newaxis]
  return img

def squeeze_dims(img, ndim):
  while img.ndim > ndim:
    img = img[..., 0]
  return img
