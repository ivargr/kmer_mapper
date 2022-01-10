import logging
import sys
logging.basicConfig(level=logging.INFO)
import numpy as np
import time
from kmer_mapper import map_kmers_to_graph_index
from npstructures import Counter
from graph_kmer_index import KmerIndex
from kmer_mapper.parser import BufferedNumpyParser, OneLineFastaBuffer2Bit
from kmer_mapper.kmers import KmerHash, TwoBitHash
from kmer_mapper.util import log_memory_usage_now

def get_kmer_hashes():
    parser = BufferedNumpyParser.from_filename(sys.argv[2], 1000000 * 130)
    chunks = parser.get_chunks()
    raw_chunk = next(chunks)
    sequence_chunk = raw_chunk.get_sequences()

    t = time.perf_counter()
    hashes = TwoBitHash(k=31).get_kmer_hashes(sequence_chunk)
    logging.info("Size of hashes (GB): %.3f" % (hashes.nbytes / 1000000000))
    logging.info("Time spent to get %d kmer hashes: %.3f" % (len(hashes), time.perf_counter() - t))

    return hashes

def map_with_counter(kmers, index):
    return index.count(kmers)


def map_with_cython(kmers, index):
    max_node_id = 1000000
    return map_kmers_to_graph_index(index, max_node_id, kmers.astype(np.uint64))

kmers = get_kmer_hashes().astype(np.int64)

logging.info("Reading kmer index")
kmer_index = KmerIndex.from_file(sys.argv[1])
kmer_index.convert_to_int32()
index_kmers = np.unique(kmer_index._kmers).astype(np.int64)
logging.info("Making counter")
counter = Counter(index_kmers)
#kmers = np.concatenate([index_kmers for i in range(100)])



logging.info("Getting kmers")

#kmers = np.random.randint(0, 4**31, 96000000, dtype=np.int64)



for i in range(0, 3):
    #for function, index in [(map_with_counter, counter), (map_with_cython, kmer_index)]:
    for function, index in [(map_with_counter, counter)]:
        t = time.perf_counter()
        function(kmers, index)
        print(str(function), time.perf_counter()-t)
        log_memory_usage_now()






