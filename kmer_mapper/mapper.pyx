#cython: language_level=3
from libc.stdio cimport *
import numpy as np
cimport numpy as np
import logging
import time
cimport cython
from graph_kmer_index.shared_mem import to_shared_memory, SingleSharedArray

def py_read_file(filename):
    with open(filename, "r") as f:
        return f.read()

cpdef test():
    print("test2")


cpdef count_lines_in_fasta(file_name):
    i = 0
    with open(file_name) as f:
        #data = f.read()
        for line in f:
            i += 1

    return i

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef void read_sequence_to_kmer_hashes(char* sequence, np.uint64_t[:] read_array, np.uint64_t[:] reverse_complement_read_array,
                                       int sequence_length):
    #cdef sequence_length = len(sequence) - 3  # last two bytes are \n
    cdef int i = 0
    cdef int i_reverse_complement = sequence_length
    cdef int base
    for i in range(0, sequence_length):
        base = sequence[i]
        if base == 97 or base == 65:
            read_array[i] = 0
            reverse_complement_read_array[i_reverse_complement] = 2
        elif base == 99 or base == 67:
            read_array[i] = 1
            reverse_complement_read_array[i_reverse_complement] = 3
        elif base == 116 or base == 84:
            read_array[i] = 2
            reverse_complement_read_array[i_reverse_complement] = 0
        elif base == 103 or base == 71:
            read_array[i] = 3
            reverse_complement_read_array[i_reverse_complement] = 1
        else:
            # invalid characters, like N, just keep these 0
            read_array[i] = 0
        i_reverse_complement -= 1


cdef void process_line(char* sequence, np.uint64_t[:] chunk_row, np.uint8_t[:] mask_row):
    cdef i = 0
    for i in range(0, len(sequence)-1):
        chunk_row[i] = sequence[i]
        mask_row[i] = True


def read_fasta_into_chunks(filename, chunk_size=1000, int max_read_length=150, write_to_shared_memory=False, process_reads=False):
    logging.info("Reading fasta %s" % filename)
    filename_byte_string = filename.encode("UTF-8")
    cdef char * fname = filename_byte_string

    cdef FILE * cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

    cdef char * line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    cdef i = 0
    cdef int line_length = 0

    if process_reads:
        chunk = np.zeros((chunk_size, max_read_length), dtype=np.uint64)
    else:
        chunk = np.empty(chunk_size, dtype="|S" + str(max_read_length))  # max_read_length bytes for each element

    mask = np.zeros((chunk_size, max_read_length),
                        dtype=np.bool)  # True where reads have bases (for handling short reads)

    prev_time = time.time()

    while True:
        read = getline(&line, &l, cfile)
        if read == -1:
            logging.info("No more reads")
            break

        # skip header lines
        if line[0] == 62:
            continue

        line_length = len(line)-1
        #chunk[i,line_length] = line
        if process_reads:
            chunk[i, 0:line_length] = np.frombuffer(line, dtype=np.uint8)[:-1]  # remove \n at end
            mask[i, 0:line_length] = True
            # a bit slower:
            #process_line(line, chunk[i,:], mask[i,:])
        else:
            chunk[i] = line
            mask[i, 0:line_length] = True

        i += 1
        if i >= chunk_size:
            if write_to_shared_memory:
                name1 = "shared_array_" + str(np.random.random())
                name2 = "shared_array_" + str(np.random.random())
                logging.info("Yielding chunk with %d reads in shared memory %s" % (chunk.shape[0], name1))
                to_shared_memory(SingleSharedArray(chunk), name1)
                to_shared_memory(SingleSharedArray(mask), name2)
                yield name1, name2
            else:
                logging.info("Yielding %d reads directly" % (len(chunk)))
                yield chunk, mask
            logging.info("Took %.3f sec to read %d reads into matrix" % (time.time()-prev_time, chunk_size))
            prev_time = time.time()


            i = 0
            if process_reads:
                chunk = np.zeros((chunk_size, max_read_length), dtype=np.uint64)
            else:
                chunk = np.empty(chunk_size, dtype="|S" + str(max_read_length))

            mask = np.zeros((chunk_size, max_read_length), dtype=np.bool)  # True where reads have bases (for handling short reads)

    if i > 0:
        logging.info("Yielding %d reads in the end" % i)
        if write_to_shared_memory:
            name1 = "shared_array_" + str(np.random.random())
            name2 = "shared_array_" + str(np.random.random())
            to_shared_memory(SingleSharedArray(chunk[0:i]), name1)
            to_shared_memory(SingleSharedArray(mask[0:i]), name2)
            yield name1, name2
        else:
            yield chunk[0:i], mask[0:i]

    fclose(cfile)



@cython.boundscheck(False)
@cython.wraparound(False)
def map_kmers_to_graph_index(index, int max_node_id, np.uint64_t[:] kmers, int max_index_lookup_frequency=1000):

    t = time.perf_counter()
    # index arrays
    cdef np.int64_t[:] hashes_to_index = index._hashes_to_index
    cdef np.uint32_t[:] n_kmers = index._n_kmers
    cdef np.uint32_t[:] nodes = index._nodes
    cdef np.uint64_t[:] index_kmers = index._kmers
    cdef np.uint16_t[:] index_frequencies = index._frequencies.data

    cdef int n_local_hits
    cdef unsigned long index_position

    cdef int l, j, i
    cdef long hash
    cdef int modulo = index._modulo
    #logging.info("Hash modulo is %d. Max index lookup frequency is %d." % (modulo, max_index_lookup_frequency))

    cdef np.ndarray[np.uint32_t] node_counts = np.zeros(max_node_id+1, dtype=np.uint32)
    cdef np.uint32_t[:] node_counts_view = node_counts

    t = time.perf_counter()
    cdef np.uint64_t[:] kmer_hashes = np.mod(kmers, modulo)
    logging.info("Time spent taking modulo: %.4f" % (time.perf_counter()-t))
    cdef int n_collisions = 0
    cdef int n_kmers_mapped = 0
    cdef int n_skipped_high_frequency = 0
    cdef int n_no_index_hits = 0
    logging.info("Time initing mapper: %.7f" % (time.perf_counter()-t))
    t = time.perf_counter()
    #logging.info("Will process %d kmers" % kmers.shape[0])
    for i in range(kmers.shape[0]):
        hash = kmer_hashes[i]

        if hash == 0:
            continue

        n_local_hits = n_kmers[hash]
        if n_local_hits == 0 or n_local_hits > max_index_lookup_frequency:
            n_no_index_hits += 1


        index_position = hashes_to_index[hash]

        for j in range(n_local_hits):
            l = index_position + j
            # Check that this entry actually matches the kmer, sometimes it will not due to collision
            if index_kmers[l] != kmers[i]:
                n_collisions += 1
                continue

            if index_frequencies[l] > max_index_lookup_frequency:
                n_skipped_high_frequency += 1
                continue

            node_counts_view[nodes[l]] += 1
            n_kmers_mapped += 1

    logging.info("Time spent looking up kmers in index: %.3f" % (time.perf_counter()-t))
    #logging.info("N kmers with no index hits: %d" % n_no_index_hits)
    #logging.info("N hash collisions: %d" % n_collisions)
    #logging.info("N skipped because too high frequency: %d" % n_skipped_high_frequency)
    logging.info("N kmers mapped: %d" % n_kmers_mapped)
    return node_counts


