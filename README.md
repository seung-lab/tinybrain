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

# Method     Run Time    Rel. Perf.
# Original   1.85 sec     1.0x
# tinybrain  0.09 sec    20.6x 
# OpenCV     0.47 sec     3.9x
# Pillow     0.90 sec     2.1x
```

## Considerations: downsample_segmentation 

To be continued.
