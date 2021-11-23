import logging
logging.basicConfig(level=logging.INFO)
from kmer_mapper.mapping import get_kmer_hashes, get_kmer_hashes_numpy
import numpy as np

def test():
    reads = np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 1, 2, 3, 2]
        ], dtype=np.uint64
    )

    hashes = get_kmer_hashes(reads, k=5)
    hashes_np = get_kmer_hashes_numpy(reads, k=5)

    assert np.all(hashes == hashes_np)


test()