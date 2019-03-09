[![Build Status](https://travis-ci.org/seung-lab/tinybrain.svg?branch=master)](https://travis-ci.org/seung-lab/tinybrain) [![PyPI version](https://badge.fury.io/py/tinybrain.svg)](https://badge.fury.io/py/tinybrain)  

# tinybrain

Image pyramid generation specialized for connectomics data types and procedures. If your brain wasn't tiny before, it will be now.  

```python 
import tinybrain 

img = load_3d_em_stack()

# factors (2,2), (2,2,1), and (2,2,1,1) are on a fast path
img_pyramid = tinybrain.downsample_with_averaging(img, factor=(2,2,1), num_mips=5)

labels = load_3d_labels()
label_pyramid = tinybrain.downsample_segmentation(labels, factor=(2,2,1), num_mips=5)
```

## Installation 

```bash
pip install numpy
pip install tinybrain
```

## Motivation

Image heirarchy generation in connectomics uses a few different techniques for
visualizing data, but predominantly we create image pyramids of uint8 grayscale 
images using 2x2 average pooling and of uint8 to uint64 segmentation labels using 
2x2 mode pooling.  

It's possible to compute both of these using numpy, however as multiple packages found 
it useful to copy the downsample functions, it makes sense to formalize these functions 
into a seperate library located on PyPI.

Given the disparate circumstances that they will be used in, these functions should work 
fast as possible with low memory usage and avoid numerical issues such as integer truncation
while generating multiple mip levels.

## Considerations: downsample_with_averaging 

It's advisable to generate multiple mip levels at once rather than recursively computing
new images as for integer type images, this leads to integer truncation issues. In the common
case of 2x2x1 downsampling, a recursively computed image would lose 0.75 brightness per a 
mip level. Therefore, take advantage of the `num_mips` argument which strikes a balance
that limits integer truncation loss to once every 4 mip levels. This compromise allows
for the use of integer arithmatic and no more memory usage than 2x the input image including
the output downsamples. If you seek to eliminate the loss beyond 4 mip levels, try promoting 
the type before downsampling.  

A C++ high performance path is triggered for 2x2x1x1 downsample factors on uint8, uint16, float32, 
and float64 data types in Fortran order. Other factors, data types, and orderings are computed using a numpy pathway that is much slower and more memory intensive.


### Example Benchmark 

On a 1024x1024x100 uint8 image I ran the following code. PIL and OpenCV are actually much faster than this benchmark shows because most of the time is spent writing to the numpy array. tinybrain has a large advantage working on 3D and 4D arrays. Of course, this is a very simple benchmark and it may be possible to tune each of these approaches. On single slices, Pillow was faster than tinybrain.

```python
img = np.load("image.npy")

s = time.time()
downsample_with_averaging(img, (2,2,1))
print("Original ", time.time() - s)

s = time.time()
out = tinybrain.downsample_with_averaging(img, (2,2,1))
print("tinybrain ", time.time() - s)

s = time.time()
out = np.zeros(shape=(512,512,100))
for z in range(img.shape[2]):
  out[:,:,z] = cv2.resize(img[:,:,z], dsize=(512, 512) )
print("OpenCV ", time.time() - s)

s = time.time()
out = np.zeros(shape=(512,512,100))
for z in range(img.shape[2]):
  pilimg = Image.fromarray(img[:,:,z])
  out[:,:,z] = pilimg.resize( (512, 512) )
print("Pillow ", time.time() - s)

# Method     Run Time             Rel. Perf.
# Original   1820 ms +/- 3.73 ms    1.0x
# tinybrain    67 ms +/- 0.40 ms   27.2x 
# OpenCV      469 ms +/- 1.12 ms    3.9x
# Pillow      937 ms +/- 7.63 ms    1.9x
```

## Considerations: downsample_segmentation 

The `downsample_segmentation` function performs mode pooling operations provided the downsample factor is a power of two, including in three dimensions. If the factor is a non-power of two, striding is used. The mode pooling, which is usually what you want, is computed recursively. Mode pooling is superior to striding, but the recursive calculatioon can introduce defects at mip levels higher than 1. This may be improved in the future.  

The way the calculation is actually done uses an ensemble of several different methods. For (2,2,1,1) downsamples, a Cython fast, low memory path is selected that implements [*countless if*](https://towardsdatascience.com/countless-high-performance-2x-downsampling-of-labeled-images-using-python-and-numpy-e70ad3275589). For (4,4,1) or other 2D powers of two, the [*countless 2d*](https://towardsdatascience.com/countless-high-performance-2x-downsampling-of-labeled-images-using-python-and-numpy-e70ad3275589) algorithm is used. For (2,2,2), (4,4,4), etc, the [*dynamic countless 3d*](https://towardsdatascience.com/countless-3d-vectorized-2x-downsampling-of-labeled-volume-images-using-python-and-numpy-59d686c2f75) algorithm is used. For 2D powers of two, [*stippled countless 2d*](https://medium.com/@willsilversmith/countless-2d-inflated-2x-downsampling-of-labeled-images-holding-zero-values-as-background-4d13a7675f2d) is used if the sparse flag is enabled. For all other configurations, striding is used.  

Countless 2d paths are also fast, but use slightly more memory and time. Countless 3D is okay for (2,2,2) and (4,4,4) but will use time and memory exponential in the product of dimensions. This state of affairs could be improved by implementing a counting based algorithm in Cython/C++ for arbitrary factors that doesn't compute recursively. The countless algorithms were developed before I knew how to write Cython and package libraries. However, C++ implementations of countless are much faster than counting for computing the first mip level. In particular, an AVX2 SIMD implementation can saturate memory bandwidth.    

Documentation for the countless algorithm family is located here: https://github.com/william-silversmith/countless  


