# Results on a 2021 Macbook Pro M1
# AVG POOLING ((2048, 2048, 64), uint8)
# 2x2 1 mip: 0.695s, 3856.07 MVx/sec, N=10
# 2x2 2 mip: 0.998s, 2685.80 MVx/sec, N=10
# 2x2x2 1 mip: 0.600s, 4468.55 MVx/sec, N=10
# 2x2x2 2 mip: 0.935s, 2867.80 MVx/sec, N=10
# 2x2x2 1 mip sparse: 1.929s, 1389.47 MVx/sec, N=10
# 2x2x2 2 mip sparse: 2.128s, 1259.58 MVx/sec, N=10

# MODE POOLING RANDOM ((2048, 2048, 64), uint64)
# 2x2 1 mip: 2.533s, 1057.87 MVx/sec, N=10
# 2x2 2 mip: 2.522s, 1062.69 MVx/sec, N=10
# 2x2 1 mip sparse: 20.673s, 129.64 MVx/sec, N=10
# 2x2 2 mip sparse: 32.836s, 81.62 MVx/sec, N=10
# 2x2x2 1 mip: 7.956s, 336.85 MVx/sec, N=10
# 2x2x2 2 mip: 8.980s, 298.45 MVx/sec, N=10
# 2x2x2 1 mip sparse: 7.932s, 337.87 MVx/sec, N=10
# 2x2x2 2 mip sparse: 9.121s, 293.84 MVx/sec, N=10

# MODE POOLING CONNECTOMICS ((512,512,512), uint64)
# 2x2 1 mip: 1.243s, 1078.09 MVx/sec, N=10
# 2x2 2 mip: 1.300s, 1030.90 MVx/sec, N=10
# 2x2 1 mip sparse: 9.168s, 146.15 MVx/sec, N=10
# 2x2 2 mip sparse: 19.350s, 69.25 MVx/sec, N=10
# 2x2x2 1 mip: 0.681s, 1966.74 MVx/sec, N=10
# 2x2x2 2 mip: 0.748s, 1790.60 MVx/sec, N=10
# 2x2x2 1 mip sparse: 0.656s, 2041.96 MVx/sec, N=10
# 2x2x2 2 mip sparse: 0.762s, 1758.42 MVx/sec, N=10

# Results on a 2023 Macbook Pro M3
# AVG POOLING ((2048, 2048, 64), uint8)
# 2x2 1 mip: 0.084s, 31954.51 MVx/sec, N=10
# 2x2 2 mip: 0.223s, 11995.13 MVx/sec, N=10
# 2x2x2 1 mip: 0.042s, 63958.66 MVx/sec, N=10
# 2x2x2 2 mip: 0.218s, 12303.19 MVx/sec, N=10
# 2x2x2 1 mip sparse: 0.581s, 4612.79 MVx/sec, N=10
# 2x2x2 2 mip sparse: 0.686s, 3908.00 MVx/sec, N=10
# 2x2 1 mip pillow: 10.526s, 254.61 MVx/sec, N=10
# 2x2 1 mip opencv: 5.352s, 500.79 MVx/sec, N=10
# MODE POOLING ((2048, 2048, 64), uint64)
# 2x2 1 mip: 1.030s, 2602.65 MVx/sec, N=10
# 2x2 2 mip: 1.181s, 2269.54 MVx/sec, N=10
# 2x2 1 mip sparse: 14.294s, 187.49 MVx/sec, N=10
# 2x2 2 mip sparse: 25.615s, 104.63 MVx/sec, N=10
# 2x2x2 1 mip: 3.621s, 740.07 MVx/sec, N=10
# 2x2x2 2 mip: 4.066s, 659.13 MVx/sec, N=10
# 2x2x2 1 mip sparse: 4.356s, 615.27 MVx/sec, N=10
# 2x2x2 2 mip sparse: 5.084s, 527.10 MVx/sec, N=10

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

	try:
		from PIL import Image
		s = time.time()
		out = np.zeros([1024,1024,64], dtype=np.uint8, order="F")
		for _ in range(N):
			for z in range(data.shape[2]):
			  pilimg = Image.fromarray(data[:,:,z])
			  out[:,:,z] = pilimg.resize( (1024, 1024) )
		end = time.time()
		result("2x2 1 mip pillow", end - start, data, N)
		del out
	except ImportError:
		print("Pillow not installed.")

	try:
		import cv2
		start = time.time()
		out = np.zeros(shape=(1024,1024,64))
		for _ in range(N):
			for z in range(data.shape[2]):
			  out[:,:,z] = cv2.resize(data[:,:,z], dsize=(1024, 1024) )
		end = time.time()
		result("2x2 1 mip opencv", end - start, data, N)
		del out
	except ImportError:
		print("opencv not installed.")

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
