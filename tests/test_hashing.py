import numpy as np
from bionumpy import as_sequence_array
from bionumpy.kmers import fast_hash
from bionumpy.encodings import ACTGTwoBitEncoding, ACTGEncoding

def test():
    from graph_kmer_index import letter_sequence_to_numeric
    print("")
    k = 3
    strings = ["AAAAAAAA", "TTTTTTTTT", "AAAATTTT", "TTTTAAAA"]
    sequence = as_sequence_array(strings, encoding=ACTGEncoding)
    print(sequence)
    hashes = fast_hash(sequence, k).ravel().astype(np.uint64)
    print(hashes)

    hashes_complement = ACTGTwoBitEncoding.complement(hashes) & np.uint64(4**k-1)
    print("Hashes complement")
    print(hashes_complement)

    numeric_sequences = [letter_sequence_to_numeric(s) for s in strings]
    print(numeric_sequences)

    old = [
        np.convolve(s, 4**np.arange(k), mode='valid') for s in numeric_sequences
    ]


    print("OLD")
    print(old)

    print(fast_hash(numeric_sequences[-1][::-1], k).ravel())


def test2():

    numeric_sequence = np.array(np.arange(32) % 4, dtype=np.uint8)
    #numeric_sequence = np.array([1, 1, 1, 1], dtype=np.uint8)
    k = len(numeric_sequence)-1
    hash = fast_hash(numeric_sequence, k, encoding=None)
    print(hash)

    hashes_complement = ACTGTwoBitEncoding.complement(hash) & np.uint64(4**k-1)
    print(hashes_complement)

    reverse_complement = ((numeric_sequence+2)%4)[::-1]
    hash_convolve = np.convolve(reverse_complement.astype(np.uint64), 4**np.arange(k).astype(np.uint64), mode='valid')
    print(hash_convolve)


if __name__ == "__main__":
    test2()