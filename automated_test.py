import pytest

from copy import deepcopy
import numpy as np

import tinybrain

image1x1x1 = np.array([[[[0]]]])
image1x1x1f = np.asfortranarray(image1x1x1)

image2x2x2 = np.array([ 
  [
    [ [1], [1] ],
    [ [2], [2] ],
  ], 
  [
    [ [1], [0 ] ],
    [ [0], [30] ],
  ] 
])
image2x2x2f = np.asfortranarray(image2x2x2)

image3x3x3 = np.array([ 
  [#z 0  1  2   
    [ [1], [1], [1] ], # y=0
    [ [1], [1], [1] ], # y=1      # x=0
    [ [1], [1], [1] ], # y=2
  ], 
  [
    [ [2], [2], [2] ], # y=0 
    [ [2], [2], [2] ], # y=1      # x=1
    [ [2], [2], [2] ], # y=2
  ],
  [
    [ [3], [3], [3] ], # y=0
    [ [3], [3], [3] ], # y=1      # x=2
    [ [3], [3], [3] ], # y=2
  ],
])
image3x3x3f = np.asfortranarray(image3x3x3)

image4x4x4 = np.array([ 
  [ #z 0    1    2   3 
    [ [1], [1], [1], [1] ], # y=0
    [ [1], [1], [1], [1] ], # y=1      # x=0
    [ [1], [1], [1], [1] ], # y=2
    [ [1], [1], [1], [1] ], # y=3
  ], 
  [
    [ [2], [2], [2], [2] ], # y=0
    [ [2], [2], [2], [2] ], # y=1      # x=1
    [ [2], [2], [2], [2] ], # y=2
    [ [2], [2], [2], [2] ], # y=3
  ], 
  [
    [ [3], [3], [3], [3] ], # y=0
    [ [3], [3], [3], [3] ], # y=1      # x=2
    [ [3], [3], [3], [3] ], # y=2
    [ [3], [3], [3], [3] ], # y=3
  ],
  [
    [ [4], [4], [4], [4] ], # y=0
    [ [4], [4], [4], [4] ], # y=1      # x=3
    [ [4], [4], [4], [4] ], # y=2
    [ [4], [4], [4], [4] ], # y=3
  ], 
])
image4x4x4f = np.asfortranarray(image4x4x4)


def test_even_odd2d():
  evenimg = tinybrain.downsample.odd_to_even2d(image2x2x2)
  assert np.array_equal(evenimg, image2x2x2)

  evenimg = tinybrain.downsample.odd_to_even2d(image2x2x2f)
  assert np.array_equal(evenimg, image2x2x2f)

  oddimg = tinybrain.downsample.odd_to_even2d(image1x1x1).astype(int)
  oddimgf = tinybrain.downsample.odd_to_even2d(image1x1x1f).astype(int)
  
  ans1x1x1 = np.array([
    [
      [ [0] ],
      [ [0] ], 
    ],
    [
      [ [0] ],
      [ [0] ] 
    ]
  ])

  assert np.array_equal(oddimg, ans1x1x1)
  assert np.array_equal(oddimgf, ans1x1x1)

  oddimg = tinybrain.downsample.odd_to_even2d(image3x3x3)
  oddimgf = tinybrain.downsample.odd_to_even2d(image3x3x3f)

  ans3x3x3 = np.array([
    [
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
    ],
    [
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
      [ [1], [1], [1] ], 
    ],
    [
      [ [2], [2], [2] ],
      [ [2], [2], [2] ],
      [ [2], [2], [2] ],
      [ [2], [2], [2] ],
    ],
    [
      [ [3], [3], [3] ],
      [ [3], [3], [3] ],
      [ [3], [3], [3] ],
      [ [3], [3], [3] ],
    ]
  ])

  assert np.array_equal(oddimg, ans3x3x3)
  assert np.array_equal(oddimgf, ans3x3x3)

@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32, np.float64))
def test_accelerated_vs_numpy_avg_pooling_2x2(dtype):
  image = np.random.randint(0,255, size=(512, 512, 6), dtype=np.uint8).astype(dtype)
  imagef = np.asfortranarray(image)

  accimg = tinybrain.accelerated.average_pooling_2x2(imagef) 
  npimg = tinybrain.downsample.downsample_with_averaging_numpy(imagef, (2,2,1))
  assert np.all(accimg == npimg)

  # There are slight differences in how the accelerated version and 
  # the numpy version handle the edge so we only compare a nice 
  # even power of two where there's no edge. We also can't do iterated
  # downsamples of the (naked) numpy version because it will result in
  # integer truncation. We can't compare above mip 4 because the accelerated
  # version will exhibit integer truncation.

  mips = tinybrain.downsample_with_averaging(imagef, (2,2,1), num_mips=4)
  npimg = tinybrain.downsample.downsample_with_averaging_numpy(imagef, (16,16,1))
  
  assert np.all(mips[-1] == npimg)

@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("sx", (6,7,1024,1025))
@pytest.mark.parametrize("sy", (6,7,1024,1025))
@pytest.mark.parametrize("sz", (4,5,32,33))
def test_accelerated_vs_numpy_avg_pooling_2x2x2(dtype, sx, sy, sz):
  image = np.random.randint(0,255, size=(sx, sy, sz), dtype=np.uint8).astype(dtype)
  imagef = np.asfortranarray(image)

  accimg = tinybrain.accelerated.average_pooling_2x2x2(imagef) 
  npimg = tinybrain.downsample.downsample_with_averaging_numpy(imagef, (2,2,2))
  assert np.all(accimg == npimg)

  # There are slight differences in how the accelerated version and 
  # the numpy version handle the edge so we only compare a nice 
  # even power of two where there's no edge. We also can't do iterated
  # downsamples of the (naked) numpy version because it will result in
  # integer truncation. We can't compare above mip 4 because the accelerated
  # version will exhibit integer truncation.

  if ( x % 2 for x in (sx,sy,sz)) == (0,0,0):
    mips = tinybrain.downsample_with_averaging(imagef, (2,2,2), num_mips=2)
    npimg = tinybrain.downsample.downsample_with_averaging_numpy(imagef, (4,4,4))
    assert np.all(mips[-1] == npimg)

def test_accelerated_vs_numpy_mode_pooling():
  image = np.random.randint(0,255, size=(512, 512, 6, 1), dtype=np.uint8)

  accimg = tinybrain.accelerated.mode_pooling_2x2(image) 
  npimg = tinybrain.downsample.countless2d(image)
  assert np.all(accimg == npimg)

  # There are slight differences in how the accelerated version and 
  # the numpy version handle the edge so we only compare a nice 
  # even power of two where there's no edge. We also can't do iterated
  # downsamples of the (naked) numpy version because it will result in
  # integer truncation. We can't compare above mip 4 because the accelerated
  # version will exhibit integer truncation.

  mips = tinybrain.downsample_segmentation(image, (2,2,1), num_mips=4)
  npimg = tinybrain.downsample.downsample_segmentation_2d(image, (16,16,1), sparse=False)
  
  assert np.all(mips[-1] == npimg)

@pytest.mark.parametrize('order', ('C', 'F'))
def test_downsample_segmentation_4x_z(order):
  case1 = np.array([ [ 0, 1 ], [ 2, 3 ] ]).reshape((2,2,1,1), order=order) # all different
  case2 = np.array([ [ 0, 0 ], [ 2, 3 ] ]).reshape((2,2,1,1), order=order) # two are same
  case3 = np.array([ [ 1, 1 ], [ 2, 2 ] ]).reshape((2,2,1,1), order=order) # two groups are same
  case4 = np.array([ [ 1, 2 ], [ 2, 2 ] ]).reshape((2,2,1,1), order=order) # 3 are the same
  case5 = np.array([ [ 5, 5 ], [ 5, 5 ] ]).reshape((2,2,1,1), order=order) # all are the same

  is_255_handled = np.array([ [ 255, 255 ], [ 1, 2 ] ], dtype=np.uint8).reshape((2,2,1,1))

  test = lambda case: tinybrain.downsample.countless2d(case)[0][0][0][0]

  assert test(case1) == 3 # d
  assert test(case2) == 0 # a==b
  assert test(case3) == 1 # a==b
  assert test(case4) == 2 # b==c
  assert test(case5) == 5 # a==b

  assert test(is_255_handled) == 255 

  assert tinybrain.downsample.countless2d(case1).dtype == case1.dtype

  #  0 0 1 3 
  #  1 1 6 3  => 1 3

  case_odd = np.array([ 
    [
      [ [1] ], 
      [ [0] ] 
    ],
    [
      [ [1] ],
      [ [6] ],
    ],
    [
      [ [3] ],
      [ [3] ],
    ],
  ]) # all are the same

  downsamplefn = tinybrain.downsample.downsample_segmentation

  result, = downsamplefn(case_odd, (2,2,1))
  assert np.array_equal(result, np.array([
    [
      [ [1] ]
    ],
    [
      [ [3] ]
    ]
  ]))

  data = np.ones(shape=(1024, 511, 62, 1), dtype=int)
  result, = downsamplefn(data, (2,2,1))
  assert result.shape == (512, 256, 62, 1)

def test_downsample_segmentation_4x_x():
  case1 = np.array([ [ 0, 1 ], [ 2, 3 ] ]).reshape((1,2,2,1)) # all different
  case2 = np.array([ [ 0, 0 ], [ 2, 3 ] ]).reshape((1,2,2,1)) # two are same
  case3 = np.array([ [ 1, 1 ], [ 2, 2 ] ]).reshape((1,2,2,1)) # two groups are same
  case4 = np.array([ [ 1, 2 ], [ 2, 2 ] ]).reshape((1,2,2,1)) # 3 are the same
  case5 = np.array([ [ 5, 5 ], [ 5, 5 ] ]).reshape((1,2,2,1)) # all are the same

  is_255_handled = np.array([ [ 255, 255 ], [ 1, 2 ] ], dtype=np.uint8).reshape((1,2,2,1))

  test = lambda case: tinybrain.downsample.downsample_segmentation(case, (1,2,2))[0][0][0][0]

  assert test(case1) == 3 # d
  assert test(case2) == 0 # a==b
  assert test(case3) == 1 # a==b
  assert test(case4) == 2 # b==c
  assert test(case5) == 5 # a==b

  assert test(is_255_handled) == 255 

  result, = tinybrain.downsample.downsample_segmentation(case1, (1,2,2))
  assert result.dtype == case1.dtype

  #  0 0 1 3 
  #  1 1 6 3  => 1 3

  case_odd = np.array([ 
    [
      [ [1], [0] ], 
      [ [1], [6] ],
      [ [3], [3] ]
    ]
  ]) # all are the same

  downsamplefn = tinybrain.downsample.downsample_segmentation

  result, = downsamplefn(case_odd, (1,2,2))

  assert np.array_equal(result, np.array([
    [
      [ [1] ],
      [ [3] ]
    ]
  ]))

  data = np.ones(shape=(1024, 62, 511, 1), dtype=int)
  result, = downsamplefn(data, (1,2,2))
  assert result.shape == (1024, 31, 256, 1)

  result, = downsamplefn(result, (1,2,2))
  assert result.shape == (1024, 16, 128, 1)

def test_downsample_max_pooling():
  for dtype in (np.int8, np.float32):
    cases = [
      np.array([ [ -1, 0 ], [ 0, 0 ] ], dtype=dtype), 
      np.array([ [ 0, 0 ], [ 0, 0 ] ], dtype=dtype), 
      np.array([ [ 0, 1 ], [ 0, 0 ] ], dtype=dtype),
      np.array([ [ 0, 1 ], [ 1, 0 ] ], dtype=dtype),
      np.array([ [ 0, 1 ], [ 0, 2 ] ], dtype=dtype)
    ]

    for i in range(len(cases)):
      case = cases[i]
      result, = tinybrain.downsample.downsample_with_max_pooling(case, (1, 1))
      assert np.all(result == cases[i])

    answers = [ 0, 0, 1, 1, 2 ]

    for i in range(len(cases)):
      case = cases[i]
      result, = tinybrain.downsample.downsample_with_max_pooling(case, (2, 2))
      assert result == answers[i]


    cast = lambda arr: np.array(arr, dtype=dtype) 

    answers = list(map(cast, [  
      [[ 0, 0 ]],
      [[ 0, 0 ]],
      [[ 0, 1 ]],
      [[ 1, 1 ]],
      [[ 0, 2 ]],
    ]))

    for i in range(len(cases)):
      case = cases[i]
      result, = tinybrain.downsample.downsample_with_max_pooling(case, (2, 1))
      assert np.all(result == answers[i])

    answers = list(map(cast, [  
      [[ 0 ], [ 0 ]],
      [[ 0 ], [ 0 ]],
      [[ 1 ], [ 0 ]],
      [[ 1 ], [ 1 ]],
      [[ 1 ], [ 2 ]],
    ]))

    for i in range(len(cases)):
      case = cases[i]
      result, = tinybrain.downsample.downsample_with_max_pooling(case, (1, 2))
      assert np.all(result == answers[i])

  result, = tinybrain.downsample.downsample_with_max_pooling(image4x4x4, (2, 2, 2))
  answer = cast([
    [
      [ [2], [2] ], # y=0    # x = 0
      [ [2], [2] ]  # y=1
    ],
    [
      [ [4], [4] ], # y = 0     # x = 1
      [ [4], [4] ]  # y = 1
    ]
  ])

  assert np.all(result == answer)

  result, = tinybrain.downsample.downsample_with_max_pooling(image4x4x4, (2, 2, 1))
  answer = cast([
    [
      [ [2], [2], [2], [2] ], # y=0    # x = 0
      [ [2], [2], [2], [2] ]  # y=1
    ],
    [
      [ [4], [4], [4], [4] ], # y = 0     # x = 1
      [ [4], [4], [4], [4] ]  # y = 1
    ]
  ])

  assert np.all(result == answer)

  result, = tinybrain.downsample.downsample_with_max_pooling(image4x4x4, (4, 2, 1))
  answer = cast([
    [
      [ [4], [4], [4], [4] ], # y = 0     # x = 1
      [ [4], [4], [4], [4] ]  # y = 1
    ]
  ])

  assert np.all(result == answer)

def test_countless3d():
  def test_all_cases(fn):
    alldifferent = [
      [
        [1,2],
        [3,4],
      ],
      [
        [5,6],
        [7,8]
      ]
    ]
    allsame = [
      [
        [1,1],
        [1,1],
      ],
      [
        [1,1],
        [1,1]
      ]
    ]

    assert fn(np.array(alldifferent))[0,0,0] in list(range(1,9))
    assert fn(np.array(allsame)) == [[[1]]]

    twosame = deepcopy(alldifferent)
    twosame[1][1][0] = 2

    assert fn(np.array(twosame)) == [[[2]]]

    threemixed = [
      [
        [3,3],
        [1,2],
      ],
      [
        [2,4],
        [4,3]
      ]
    ]
    assert fn(np.array(threemixed)) == [[[3]]]

    foursame = [
      [
        [4,4],
        [1,2],
      ],
      [
        [2,4],
        [4,3]
      ]
    ]

    assert fn(np.array(foursame)) == [[[4]]]

    fivesame = [
      [
        [5,4],
        [5,5],
      ],
      [
        [2,4],
        [5,5]
      ]
    ]

    assert fn(np.array(fivesame)) == [[[5]]]


  test_all_cases(tinybrain.downsample.countless3d)
  test_all_cases(lambda case: tinybrain.downsample.downsample_segmentation(case, (2,2,2))[0])

  odddimension = np.array([
    [
      [5,4],
      [5,5],
    ],
    [
      [2,4],
      [5,5]
    ],
    [
      [2,4],
      [5,5]
    ]
  ])

  # this should use striding
  res, = tinybrain.downsample.downsample_segmentation(odddimension, (2,2,2))
  assert res.shape == (2, 1, 1)

  odddimension = np.array([
    [
      [5,4],
      [5,5],
    ],
    [
      [2,4],
      [5,5]
    ]
  ], dtype=np.float32)
  # this should use striding
  res, = tinybrain.downsample.downsample_segmentation(odddimension, (2,2,2))
  assert res.dtype == np.float32
  assert res.shape == (1, 1, 1)

@pytest.mark.parametrize('dtype', (np.uint8, np.uint16))
def test_sparse_2x2x2_mode_downsampling(dtype):
  ones = np.array([
    [
      [1,1],
      [1,1],
    ],
    [
      [1,1],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(ones, sparse=True)[0]
  assert res[0][0][0] == 1
  assert res.shape == (1,1,1)

  test2 = np.array([
    [
      [1,1],
      [2,1],
    ],
    [
      [1,1],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(test2, sparse=True)[0]
  assert res[0][0][0] == 1
  assert res.shape == (1,1,1)

  test3 = np.array([
    [
      [1,1],
      [2,2],
    ],
    [
      [2,2],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(test3, sparse=True)[0]
  assert res[0][0][0] == 1
  assert res.shape == (1,1,1)


  test4 = np.array([
    [
      [1,1],
      [2,2],
    ],
    [
      [2,2],
      [1,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(test4, sparse=True)[0]
  assert res[0][0][0] == 2
  assert res.shape == (1,1,1)

  test5 = np.array([
    [
      [0,0],
      [2,0],
    ],
    [
      [0,0],
      [0,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(test5, sparse=True)[0]
  assert res[0][0][0] == 2
  assert res.shape == (1,1,1)

  test6 = np.array([
    [
      [0,0],
      [0,0],
    ],
    [
      [0,0],
      [0,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.mode_pooling_2x2x2(test6, sparse=True)[0]
  assert res[0][0][0] == 0
  assert res.shape == (1,1,1)

  # test7 = np.array([
  #   [
  #     [1,0,1],
  #     [0,0,1],
  #   ],
  #   [
  #     [0,0,1],
  #     [0,0,1],
  #   ],
  # ], dtype=dtype, order='F')

  # res = tinybrain.accelerated.mode_pooling_2x2x2(test7, sparse=True)[0]
  # assert np.all(res == [[[1,1]]])
  # assert res.shape == (1,1,2)

@pytest.mark.parametrize('dtype', (np.uint8, np.uint16))
def test_sparse_2x2x2_avg_downsampling(dtype):
  ones = np.array([
    [
      [1,1],
      [1,1],
    ],
    [
      [1,1],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(ones, sparse=True)[0]
  assert res[0][0][0] == 1
  assert res.shape == (1,1,1)

  test2 = np.array([
    [
      [1,1],
      [2,1],
    ],
    [
      [1,1],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test2, sparse=True)[0]
  assert res[0][0][0] == 1
  assert res.shape == (1,1,1)

  test3 = np.array([
    [
      [7,1],
      [2,2],
    ],
    [
      [2,2],
      [1,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test3, sparse=True)[0]
  assert res[0][0][0] == 2 # 18 / 8
  assert res.shape == (1,1,1)


  test4 = np.array([
    [
      [1,1],
      [2,2],
    ],
    [
      [2,2],
      [1,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test4, sparse=True)[0]
  assert res[0][0][0] == 1 # 11/7
  assert res.shape == (1,1,1)

  test5 = np.array([
    [
      [0,0],
      [2,0],
    ],
    [
      [0,0],
      [0,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test5, sparse=True)[0]
  assert res[0][0][0] == 2
  assert res.shape == (1,1,1)

  test6 = np.array([
    [
      [0,0],
      [0,0],
    ],
    [
      [0,0],
      [0,0],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test6, sparse=True)[0]
  assert res[0][0][0] == 0
  assert res.shape == (1,1,1)

  test7 = np.array([
    [
      [1,0,1],
      [0,0,1],
    ],
    [
      [0,0,1],
      [0,0,1],
    ],
  ], dtype=dtype, order='F')

  res = tinybrain.accelerated.average_pooling_2x2x2(test7, sparse=True)[0]
  assert np.all(res == [[[1,1]]])
  assert res.shape == (1,1,2)


def test_stippled_countless2d():
  a = np.array([ [ 1, 2 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  b = np.array([ [ 0, 2 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  c = np.array([ [ 1, 0 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  d = np.array([ [ 1, 2 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  e = np.array([ [ 1, 2 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  f = np.array([ [ 0, 0 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  g = np.array([ [ 0, 2 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  h = np.array([ [ 0, 2 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  i = np.array([ [ 1, 0 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  j = np.array([ [ 1, 2 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  k = np.array([ [ 1, 0 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  l = np.array([ [ 1, 0 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  m = np.array([ [ 0, 2 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  n = np.array([ [ 0, 0 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  o = np.array([ [ 0, 0 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  z = np.array([ [ 0, 0 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 

  def test(data):
    return tinybrain.downsample.downsample_segmentation(data, (2,2,1), sparse=True)[0]

  # Note: We only tested non-matching cases above,
  # cases f,g,h,i,j,k prove their duals work as well
  # b/c if two pixels are black, either one can be chosen
  # if they are different or the same.

  assert test(a) == [[[[4]]]] 
  assert test(b) == [[[[4]]]] 
  assert test(c) == [[[[4]]]] 
  assert test(d) == [[[[4]]]] 
  assert test(e) == [[[[1]]]] 
  assert test(f) == [[[[4]]]]  
  assert test(g) == [[[[4]]]]  
  assert test(h) == [[[[2]]]]  
  assert test(i) == [[[[4]]]] 
  assert test(j) == [[[[1]]]]  
  assert test(k) == [[[[1]]]]  
  assert test(l) == [[[[1]]]]  
  assert test(m) == [[[[2]]]]  
  assert test(n) == [[[[3]]]]  
  assert test(o) == [[[[4]]]]  
  assert test(z) == [[[[0]]]]  

  bc = np.array([ [ 0, 2 ], [ 2, 4 ] ]).reshape((2,2,1,1)) 
  bd = np.array([ [ 0, 2 ], [ 3, 2 ] ]).reshape((2,2,1,1)) 
  cd = np.array([ [ 0, 2 ], [ 3, 3 ] ]).reshape((2,2,1,1)) 
  
  assert test(bc) == [[[[2]]]]
  assert test(bd) == [[[[2]]]]
  assert test(cd) == [[[[3]]]]

  ab = np.array([ [ 1, 1 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  ac = np.array([ [ 1, 2 ], [ 1, 0 ] ]).reshape((2,2,1,1)) 
  ad = np.array([ [ 1, 0 ], [ 3, 1 ] ]).reshape((2,2,1,1)) 

  assert test(ab) == [[[[1]]]]
  assert test(ac) == [[[[1]]]]
  assert test(ad) == [[[[1]]]]

def test_minimum_size():
  labels = np.zeros((100,2,100), order="F")
  out_mips = tinybrain.downsample_with_averaging(labels, factor=(2,2,1), num_mips=1)
  assert out_mips[0].shape == (50,1,100)

  try:
    labels = np.zeros((100,1,100), order="F")
    out_mips = tinybrain.downsample_with_averaging(labels, factor=(2,2,1), num_mips=1)
    assert False
  except ValueError:
    pass

  labels = np.zeros((2,100,100), order="F")
  out_mips = tinybrain.downsample_with_averaging(labels, factor=(2,2,1), num_mips=1)
  assert out_mips[0].shape == (1,50,100)

  try:
    labels = np.zeros((1,100,100), order="F")
    out_mips = tinybrain.downsample_with_averaging(labels, factor=(2,2,1), num_mips=1)
    assert False
  except ValueError:
    pass

  labels = np.zeros((100,2,100), order="F")
  out_mips = tinybrain.downsample_segmentation(labels, factor=(2,2,1), num_mips=1)
  assert out_mips[0].shape == (50,1,100)

  labels = np.zeros((100,1,100), order="F")
  out_mips = tinybrain.downsample_segmentation(labels, factor=(2,2,1), num_mips=1)
  assert out_mips[0].shape == (50,1,100)  






