import time
import numpy as np
from kmer_mapper.gpu_counter import GpuCounter
import cupy as cp
import bionumpy as bnp
import pytest


@pytest.mark.skip()
def test_bionumpy_numpy():
    file = bnp.open("test.fa", buffer_type=bnp.TwoLineFastaBuffer)
    for i, chunk in enumerate(file.read_chunks(min_chunk_size=5000000)):
        if i > 10:
            break

    return


@pytest.mark.skip()
def test_bionumpy_cupy():
    bnp.set_backend(cp)
    file = bnp.open("test.fa", buffer_type=bnp.TwoLineFastaBuffer)
    for i, chunk in enumerate(file.read_chunks(min_chunk_size=5000000)):
        if i > 10:
            break

    return


@pytest.mark.skip()
def test_time():
    funcs = [test_bionumpy_numpy, test_bionumpy_cupy]
    for func in funcs:
        t0 = time.perf_counter()
        func()
        print(func, time.perf_counter()-t0)


@pytest.mark.skip()
def test_count():
    kmers = np.array([1, 2, 3])
    nodes = np.array([10, 11, 12])

    counter = GpuCounter(kmers, kmers, nodes)
    counter.initialize_cuda(2003)
    counter.count(np.array([1, 1, 1, 2, 3, 1, 3]))
    node_counts = counter.get_node_counts(15)
    assert np.all(node_counts[[10, 11, 12]] == [4, 1, 2])
    print("Passed")


if __name__ == "__main__":
    test_count()
    test_time()