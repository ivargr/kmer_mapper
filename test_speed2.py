import time
import logging
from graph_kmer_index import KmerIndex
from kmer_mapper.parser import OneLineFastaParser
import numpy as np
#from kmer_mapper.numpy_mapping import get_index_positions

from shared_memory_wrapper import to_shared_memory, from_shared_memory

def get_kmers():
    fasta_parser = OneLineFastaParser("tests/hg002_simulated_reads_15x.fa", 500000 * 150 // 3)
    reads = fasta_parser.parse(as_shared_memory_object=False)
    for sequence_chunk in reads:
        t = time.perf_counter()
        hashes, reverse_complement_hashes, mask = KmerHash(k=31).get_kmer_hashes(sequence_chunk)
        return hashes

def get_index_positions(kmers, index):
    t = time.perf_counter()
    kmer_hashes = kmers % index._modulo
    from_indexes = index._hashes_to_index[kmer_hashes]
    to_indexes = from_indexes + index._n_kmers[kmer_hashes]
    logging.info("Took %.5f sec to get indexes" % (time.perf_counter()-t))

    return from_indexes, to_indexes


def test(index, max_node_id, kmers):
    t = time.perf_counter()
    index_positions = get_index_positions(kmers, index)
    print(time.perf_counter()-t)



index = KmerIndex.from_file("tests/kmer_index_only_variants.npz")
#kmers = index._kmers[0:1000000]
kmers = get_kmers()
logging.info("N kmers: %d" % len(kmers))
kmers2 = kmers.copy()[::-1]
kmers3 = (kmers2+100) % index._modulo
to_shared_memory(index, "test")
index2 = from_shared_memory(KmerIndex, "test")

#index2._hashes_to_index = index2._hashes_to_index.copy()
index2._n_kmers = index2._n_kmers.copy()

get_index_positions(kmers, index)
get_index_positions(kmers, index2)
get_index_positions(kmers, index)
get_index_positions(kmers, index2)
get_index_positions(kmers3, index)
get_index_positions(kmers3, index2)
