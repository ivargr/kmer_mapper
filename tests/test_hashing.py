import time

import numpy as np
from bionumpy.sequence import get_kmers
from bionumpy.encodings.alphabet_encoding import ACTGEncoding
from bionumpy.encodings._legacy_encodings import ACTGTwoBitEncoding
import bionumpy as bnp
from kmer_mapper.util import get_kmer_hashes_from_chunk_sequence


def _test2():

    numeric_sequence = np.array(np.arange(35) % 4, dtype=np.uint8)
    #numeric_sequence = np.array([1, 1, 1, 1], dtype=np.uint8)
    k = 31
    hash = get_kmers(numeric_sequence, k).ravel()
    print(hash)

    hashes_complement = ACTGTwoBitEncoding.complement(hash) & np.uint64(4**k-1)
    print("----")
    print(hashes_complement)

    reverse_complement = ((numeric_sequence+2)%4)[::-1]
    hash_convolve = np.convolve(reverse_complement.astype(np.uint64), 4**np.arange(k).astype(np.uint64), mode='valid')
    print("----")
    print(hash_convolve[::-1])



def new():
    k = 31
    import bionumpy as bnp
    #file = bnp.open("test.fa")
    file = bnp.open("test.fa")
    for chunk in file.read_chunks(min_chunk_size=150000000):

        hash = get_kmer_hashes_from_chunk_sequence(chunk.sequence, k)
        #hash = bnp.sequence.get_kmers(bnp.as_encoded_array(chunk.sequence, ACTGEncoding), k).ravel().raw()
        # needed for same result
        #hash = ACTGTwoBitEncoding.complement(hash) & np.uint64(4 ** k - 1)
        print(hash)
        return hash


def test():
    hashes = []
    for func in [new]:
        t0 = time.perf_counter()
        result = func()
        hashes.append(result)
        print(func, time.perf_counter()-t0)





if __name__ == "__main__":
    test()