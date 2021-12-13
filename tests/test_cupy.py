import time
import numpy as np
import cupy as cp

def simple_benchmark():
    k = 31
    kmers = np.random.randint(0, 4, 120 * 500000, dtype=np.uint64)
    power_array = np.power(4, np.arange(0, k, dtype=np.uint64))

    t = time.perf_counter()
    hashes = np.convolve(kmers, power_array)
    print(hashes)
    print(time.perf_counter()-t)

    t = time.perf_counter()
    cp_kmers = cp.array(kmers)
    cp_power = cp.array(power_array)
    print(time.perf_counter()-t)
    t = time.perf_counter()
    hashes = cp.convolve(cp_kmers, cp_power)
    print(hashes)
    print(time.perf_counter()-t)