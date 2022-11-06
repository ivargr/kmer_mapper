import time

import numpy as np
from bionumpy import as_sequence_array
from bionumpy.kmers import fast_hash
from bionumpy.encodings import ACTGTwoBitEncoding, ACTGEncoding
import bionumpy as bnp
from kmer_mapper.kmers import TwoBitHash
from kmer_mapper.parser import BufferedNumpyParser, OneLineFastaBuffer2Bit


def _test2():

    numeric_sequence = np.array(np.arange(35) % 4, dtype=np.uint8)
    #numeric_sequence = np.array([1, 1, 1, 1], dtype=np.uint8)
    k = 31
    hash = fast_hash(numeric_sequence, k, encoding=None).ravel()
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
        hash = fast_hash(sequence, k).ravel()
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


def new():
    k = 3
    import bionumpy as bnp
    #file = bnp.open("test.fa")
    file = bnp.open("single_read.fa", buffer_type=bnp.TwoLineFastaBuffer)
    for chunk in file.read_chunks(150000000):
        hash = fast_hash(chunk.sequence, k).ravel()
        # needed for same result
        print(hash)
        hash = ACTGTwoBitEncoding.complement(hash) & np.uint64(4 ** k - 1)
        print(hash)


def _test():
    for func in [new]:
        t0 = time.perf_counter()
        func()
        print(func, time.perf_counter()-t0)




if __name__ == "__main__":
    _test()