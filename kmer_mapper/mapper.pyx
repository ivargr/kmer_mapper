#cython: language_level=3
from libc.stdio cimport *
import numpy as np
cimport numpy as np
import logging
import time
cimport cython
from graph_kmer_index.shared_mem import to_shared_memory, SingleSharedArray
from kmer_mapper.util import log_memory_usage_now
import gc
from libc.stdlib cimport free

@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def map_kmers_to_graph_index(index, int max_node_id, np.uint64_t[::1] kmers, int max_index_lookup_frequency=1000):
    logging.debug("Starting map_kmers_to_graph_index")
    t = time.perf_counter()
    # index arrays
    cdef np.int32_t[::1] hashes_to_index = index._hashes_to_index
    #cdef np.ndarray[np.int32_t] hashes_to_index = index._hashes_to_index
    cdef np.int32_t[::1] n_kmers = index._n_kmers
    #cdef np.ndarray[np.uint32_t] n_kmers = index._n_kmers
    cdef np.int32_t[::1] nodes = index._nodes
    cdef np.uint64_t[::1] index_kmers = index._kmers
    cdef np.uint16_t[::1] index_frequencies = index._frequencies
    cdef unsigned long modulo = index._modulo

    cdef int n_local_hits
    cdef int index_position

    cdef int i = 0
    cdef int l = 0
    cdef int j = 0
    cdef np.ndarray[np.uint32_t] node_counts = np.zeros(max_node_id+1, dtype=np.uint32)
    cdef np.uint32_t[::1] node_counts_view = node_counts


    t = time.perf_counter()
    #cdef np.uint64_t[::1] kmer_hashes = np.mod(kmers, modulo)
    #logging.info("Time spent taking modulo: %.4f" % (time.perf_counter()-t))
    cdef int n_collisions = 0
    cdef int n_kmers_mapped = 0
    cdef int n_skipped_high_frequency = 0
    cdef int n_no_index_hits = 0
    cdef unsigned long kmerhash
    t = time.perf_counter()
    #logging.info("Will process %d kmers" % kmers.shape[0])

    #cdef np.uint32_t[::1] n_local_hits_array = n_kmers[kmer_hashes]
    #cdef np.int64_t[::1] index_position_array = hashes_to_index[kmer_hashes]

    #log_memory_usage_now("starting for loop cython")
    for i in range(kmers.shape[0]):
        kmerhash = kmers[i] % modulo  # kmer_hashes[i]
        n_local_hits = n_kmers[kmerhash]
        index_position = hashes_to_index[kmerhash]
        l = index_position
        for j in range(n_local_hits):
            # Check that this entry actually matches the kmer, sometimes it will not due to collision
            if index_kmers[l] != kmers[i]:
                l += 1
                continue

            if index_frequencies[l] > max_index_lookup_frequency:
                l += 1
                continue

            node_counts[nodes[l]] += 1
            l += 1

    logging.debug("Time spent looking up kmers in index: %.3f" % (time.perf_counter()-t))
    return node_counts
