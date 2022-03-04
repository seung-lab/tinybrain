# Results on a 2021 Macbook Pro M1
# AVG POOLING ((2048, 2048, 64), uint8)
# 2x2 1 mip: 0.693s, 3868.58 MVx/sec, N=10
# 2x2 2 mip: 0.996s, 2691.03 MVx/sec, N=10
# 2x2x2 1 mip: 0.600s, 4466.76 MVx/sec, N=10
# 2x2x2 2 mip: 0.939s, 2855.51 MVx/sec, N=10
# 2x2x2 1 mip sparse: 1.914s, 1400.17 MVx/sec, N=10
# 2x2x2 2 mip sparse: 2.110s, 1270.18 MVx/sec, N=10
# MODE POOLING ((2048, 2048, 64), uint64)
# 2x2 1 mip: 9.830s, 272.63 MVx/sec, N=10
# 2x2 2 mip: 16.274s, 164.68 MVx/sec, N=10
# 2x2 1 mip sparse: 19.389s, 138.22 MVx/sec, N=10
# 2x2 2 mip sparse: 32.603s, 82.20 MVx/sec, N=10
# 2x2x2 1 mip: 7.935s, 337.74 MVx/sec, N=10
# 2x2x2 2 mip: 8.951s, 299.42 MVx/sec, N=10
# 2x2x2 1 mip sparse: 603.875s, 4.44 MVx/sec, N=10
# 2x2x2 2 mip sparse: 663.919s, 4.04 MVx/sec, N=10

import numpy as np
import tinybrain
import time

def result(label, dt, data, N):
	voxels = data.size
	mvx = voxels // (10 ** 6)
	print(f"{label}: {dt:02.3f}s, {N * mvx / dt:.2f} MVx/sec, N={N}")


def benchmark_avg_pooling():
	data = np.random.randint(0,255, size=(2048,2048,64), dtype=np.uint8)
	data = np.asfortranarray(data)
	N = 10

	print(f"AVG POOLING ({data.shape}, {data.dtype})")

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,1), num_mips=1)
	end = time.time()
	result("2x2 1 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,1), num_mips=2)
	end = time.time()
	result("2x2 2 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,2), num_mips=1)
	end = time.time()
	result("2x2x2 1 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,2), num_mips=2)
	end = time.time()
	result("2x2x2 2 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,2), num_mips=1, sparse=True)
	end = time.time()
	result("2x2x2 1 mip sparse", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_with_averaging(data, (2,2,2), num_mips=2, sparse=True)
	end = time.time()
	result("2x2x2 2 mip sparse", end - start, data, N)

def benchmark_mode_pooling():
	data = np.random.randint(1000,1255, size=(2048,2048,64), dtype=np.uint64)
	data = np.asfortranarray(data)
	N = 10

	print(f"MODE POOLING ({data.shape}, {data.dtype})")

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,1), num_mips=1)
	end = time.time()
	result("2x2 1 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,1), num_mips=2)
	end = time.time()
	result("2x2 2 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,1), num_mips=1, sparse=True)
	end = time.time()
	result("2x2 1 mip sparse", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,1), num_mips=2, sparse=True)
	end = time.time()
	result("2x2 2 mip sparse", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,2), num_mips=1)
	end = time.time()
	result("2x2x2 1 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,2), num_mips=2)
	end = time.time()
	result("2x2x2 2 mip", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,2), num_mips=1, sparse=True)
	end = time.time()
	result("2x2x2 1 mip sparse", end - start, data, N)

	start = time.time()
	for _ in range(N):
		tinybrain.downsample_segmentation(data, (2,2,2), num_mips=2, sparse=True)
	end = time.time()
	result("2x2x2 2 mip sparse", end - start, data, N)

benchmark_avg_pooling()
benchmark_mode_pooling()
