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
