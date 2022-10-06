import logging
import sys
logging.basicConfig(level=logging.INFO)
import numpy as np
import time
from kmer_mapper.mapper import map_kmers_to_graph_index
from npstructures import Counter
from graph_kmer_index import KmerIndex, CounterKmerIndex
from kmer_mapper.parser import BufferedNumpyParser, OneLineFastaBuffer2Bit
from kmer_mapper.kmers import KmerHash, TwoBitHash
from kmer_mapper.util import log_memory_usage_now
from shared_memory_wrapper import from_file, to_shared_memory, from_shared_memory

def get_kmer_hashes():
    parser = BufferedNumpyParser.from_filename(sys.argv[2], 1250000 * 130)
    chunks = parser.get_chunks()
    reads = (to_shared_memory(chunk) for chunk in chunks)

    return reads
    raw_chunk = next(chunks)
    sequence_chunk = raw_chunk.get_sequences()


    return hashes

def map_with_counter(kmers, index):
    return index.count_kmers(kmers)


def _map_with_cython(reads, index):
    max_node_id = 83559391
    
    raw_chunk = from_shared_memory(OneLineFastaBuffer2Bit, reads)
    sequence_chunk = raw_chunk.get_sequences()
    logging.info("Size of sequence chunk (GB): %.3f" % (sequence_chunk.nbytes() / 1000000000))

    t = time.perf_counter()
    hashes = TwoBitHash(k=31).get_kmer_hashes(sequence_chunk)
    logging.info("Size of hashes (GB): %.3f" % (hashes.nbytes / 1000000000))
    logging.info("Time spent to get %d kmer hashes: %.3f" % (len(hashes), time.perf_counter() - t))


def map_with_cython(hashes, index):
    return map_kmers_to_graph_index(index, index.max_node_id(), hashes, 1000)

#reads = get_kmer_hashes()

#logging.info("N kmer hashes: %d" % len(kmers))

logging.info("Reading kmer index")
kmer_index = KmerIndex.from_file(sys.argv[1])
kmer_index.convert_to_int32()
kmer_index.remove_ref_offsets()
kmers = kmer_index._kmers

print(kmers)
print("N kmers: %d" % len(kmers))

logging.info("Making counter")
#counter = Counter(index_kmers)
#counter = from_file(sys.argv[2]).counter
counter = CounterKmerIndex.from_kmer_index(kmer_index)
#kmers = np.concatenate([index_kmers for i in range(100)])


logging.info("Getting kmers")

#kmers = np.random.randint(0, 4**31, 96000000, dtype=np.int64)

import bionumpy as bnp
#file = bnp.open(sys.argv[3])
#chunks = file.read_chunks()


for _ in range(3):
    for function, index in [(map_with_counter, counter), (map_with_cython, kmer_index)]:
    #for function, index in [(map_with_cython, kmer_index2)]:
        t = time.perf_counter()
        function(kmers, index)
        print(str(function), time.perf_counter()-t)






