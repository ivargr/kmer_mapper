import time

import numpy as np
from bionumpy.sequence import get_kmers
from bionumpy.encodings.alphabet_encoding import ACTGEncoding
from bionumpy.encodings._legacy_encodings import ACTGTwoBitEncoding
import bionumpy as bnp
from kmer_mapper.kmers import TwoBitHash
from kmer_mapper.parser import BufferedNumpyParser, OneLineFastaBuffer2Bit
from kmer_mapper.command_line_interface import get_kmer_hashes_from_chunk_sequence


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


def _test3():
    k = 31
    file = "test.fa"

    # old with parsing
    parser = BufferedNumpyParser.from_filename(file, 150000000)
    buffer_type = OneLineFastaBuffer2Bit
    chunks = parser.get_chunks()
    t = time.perf_counter()
    for chunk in chunks:
        print("Time reading: ", time.perf_counter()-t)
        t0 = time.perf_counter()
        sequence_chunk = chunk.get_sequences()
        print("time getting sequences: ", time.perf_counter()-t0)
        t0 = time.perf_counter()
        hashes = TwoBitHash(k=k).get_kmer_hashes(sequence_chunk)
        print("Time hashing: " , time.perf_counter()-t0)
        #print("Hashes using olde method")
        #print(hashes)

    print("Time: ", time.perf_counter()-t)

    print("")

    t = time.perf_counter()
    # new with parsing
    file = bnp.open(file)
    t0  = time.perf_counter()
    for chunk in file.read_chunks(150000000):
        print("TIMe reading from file: ", time.perf_counter()-t0)
        t0 = time.perf_counter()
        #encoded_sequence = ACTGEncoding.encode(chunk.sequence)
        #encoded_sequence = chunk.sequence.view(bnp.sequences.ASCIIText)
        sequence = chunk.sequence
        t0 = time.perf_counter()
        hash = get_kmers(sequence, k).ravel()
        print("Time hashing: ", time.perf_counter()-t0)

        t0 = time.perf_counter()
        #hash = ACTGTwoBitEncoding.complement(hash) & np.uint64(4 ** k - 1)
        print("Time rev comp: ", time.perf_counter()-t0)
        #hash = TwoBitHash(k=k).get_kmer_hashes(ACTGEncoding.encode(chunk.sequence))

        #print("----")
        #print("Hashes using new method:")
        #print(hash)

    print("Time: ", time.perf_counter()-t)




def old():
    from kmer_mapper.kmers import TwoBitHash
    from kmer_mapper.parser import BufferedNumpyParser, OneLineFastaBuffer2Bit
    parser = BufferedNumpyParser.from_filename("test.fa", 150000000)
    for chunk in parser.get_chunks():
        hashes = TwoBitHash(k=31).get_kmer_hashes(chunk.get_sequences())
        return hashes


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
    for func in [old, new]:
        t0 = time.perf_counter()
        result = func()
        hashes.append(result)
        print(func, time.perf_counter()-t0)

    print(hashes)
    assert np.all(hashes[0] == hashes[1])




if __name__ == "__main__":
    test()