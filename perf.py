# Results on a 2021 Macbook Pro M1
# AVG POOLING ((2048, 2048, 64), uint8)
# 2x2 1 mip: 0.693s, 3868.58 MVx/sec, N=10
# 2x2 2 mip: 0.996s, 2691.03 MVx/sec, N=10
# 2x2x2 1 mip: 0.600s, 4466.76 MVx/sec, N=10
# 2x2x2 2 mip: 0.939s, 2855.51 MVx/sec, N=10
# 2x2x2 1 mip sparse: 1.914s, 1400.17 MVx/sec, N=10
# 2x2x2 2 mip sparse: 2.110s, 1270.18 MVx/sec, N=10

# MODE POOLING ((2048, 2048, 64), uint64)
# 2x2 1 mip: 4.781s, 560.54 MVx/sec, N=10
# 2x2 2 mip: 5.556s, 482.34 MVx/sec, N=10
# 2x2 1 mip sparse: 19.580s, 136.87 MVx/sec, N=10
# 2x2 2 mip sparse: 33.501s, 80.00 MVx/sec, N=10
# 2x2x2 1 mip: 7.951s, 337.08 MVx/sec, N=10
# 2x2x2 2 mip: 8.973s, 298.66 MVx/sec, N=10
# 2x2x2 1 mip sparse: 562.670s, 4.76 MVx/sec, N=10
# 2x2x2 2 mip sparse: 620.148s, 4.32 MVx/sec, N=10

# MODE POOLING CONNECTOMICS ((512,512,512), uint64)
# 2x2 1 mip: 2.665s, 502.77 MVx/sec, N=10
# 2x2 2 mip: 3.116s, 430.00 MVx/sec, N=10
# 2x2 1 mip sparse: 9.280s, 144.40 MVx/sec, N=10
# 2x2 2 mip sparse: 19.599s, 68.37 MVx/sec, N=10
# 2x2x2 1 mip: 0.680s, 1969.19 MVx/sec, N=10
# 2x2x2 2 mip: 0.751s, 1783.39 MVx/sec, N=10
# 2x2x2 1 mip sparse: 186.203s, 7.20 MVx/sec, N=10
# 2x2x2 2 mip sparse: 204.116s, 6.56 MVx/sec, N=10

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
