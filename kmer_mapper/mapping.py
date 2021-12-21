import gc
import itertools
import logging
import time
import numpy as np
#from mapper import read_fasta_into_chunks, \
from mapper import map_kmers_to_graph_index
from kmer_mapper.util import remap_array
from scipy.ndimage import convolve1d
import pandas as pd
import scipy.signal
from graph_kmer_index import KmerIndex
from shared_memory_wrapper import from_shared_memory, to_shared_memory, SingleSharedArray
from shared_memory_wrapper.shared_memory import get_shared_pool, close_shared_pool
from pathos.multiprocessing import Pool
from itertools import repeat
from graph_kmer_index.collision_free_kmer_index import MinimalKmerIndex

from .hash_table import NodeCount
from .parser import OneLineFastaParser, Sequences, OneLineFastaParser2bit, BufferedNumpyParser, OneLineFastaBuffer2Bit
from .util import log_memory_usage_now
from .kmers import KmerHash, TwoBitHash


def get_reads_as_matrices(read_file_name, chunk_size=500000, max_read_length=150):
    return (chunk for chunk in read_fasta_into_chunks(read_file_name, chunk_size, max_read_length))


def convert_byte_read_array_to_int_array(reads, read_length):
    # each item is a byesequence, one byte for each base pair
    return np.frombuffer(reads, dtype=np.uint8).reshape((len(reads), read_length))


def convert_read_matrix_to_numeric(read_matrix, give_complement_base_values=False):
    # from byte values
    from_values = np.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=np.uint64)  # NB: Must be increasing
    # to internal base values for a, c, t, g
    to_values = np.array([0, 1, 3, 2, 0, 1, 3, 2], dtype=np.uint64)
    if give_complement_base_values:
        to_values = np.array([2, 3, 1, 0, 2, 3, 1, 0], dtype=np.uint64)

    return remap_array(read_matrix, from_values, to_values)


def get_kmer_hashes(numeric_read_matrix, is_complement=False, k=31):
    power_array = np.power(4, np.arange(0, k, dtype=np.uint64), dtype=np.uint64)
    if is_complement:
        power_array = np.power(4, np.arange(0, k, dtype=np.uint64)[::-1], dtype=np.uint64)

    k_half = k//2  # we don't want first k half and last k half columns
    assert numeric_read_matrix.dtype == np.uint64
    assert power_array.dtype == np.uint64

    #res = scipy.signal.fftconvolve(numeric_read_matrix, power_array, mode="valid")[:,k_half:-k_half]
    return convolve1d(numeric_read_matrix, power_array, mode="constant")[:,k_half:-k_half]

def get_kmer_hashes_numpy(numeric_read_matrix, is_complement=False, k=31):
    power_array = np.power(4, np.arange(0, k, dtype=np.uint64), dtype=np.uint64)
    if is_complement:
        power_array = np.power(4, np.arange(0, k, dtype=np.uint64)[::-1], dtype=np.uint64)

    assert numeric_read_matrix.dtype == np.uint64
    assert power_array.dtype == np.uint64

    # idea: convolve a 1d version of the 2d matrix. Use mode full and remove the first k-1 which are when power array is
    # convolved outside beginning. Reshape back to matrix and remove last k-1 columns which are overlaps between rows
    function = np.convolve
    #function = scipy.signal.convolve
    flat_numeric_read_matrix = numeric_read_matrix.flatten()
    if len(flat_numeric_read_matrix) == 0:
        logging.warning("Read matrix is empty")
        raise Exception("No reads")
        return np.zeros_like(numeric_read_matrix)

    return function(flat_numeric_read_matrix, power_array, mode="full")[k-1:].reshape(numeric_read_matrix.shape)[:,:-(k-1)]


def get_unique_kmers_and_counts(hash_matrix):
    return np.unique(hash_matrix, return_counts=True)
    unique = pd.value_counts(hash_matrix, sort=False)
    return unique.keys(), unique.values


def get_kmers_from_read_matrix(read_matrix, mask, k=31, return_only_kmers=False, include_reverse_complement=True):
    logging.info("k=%d" % k)
    t = time.time()
    numeric_reads = convert_read_matrix_to_numeric(read_matrix)
    numeric_reads_complement = (numeric_reads + 2) % 4  #convert_read_matrix_to_numeric(read_matrix, True)
    logging.info("Time to convert reads to numeric: %.3f" % (time.time()-t))
    t = time.time()

    mask = mask[:,k-1:]  # make mask for hashes (we don't want hashes for bases not from reads, which are False in original mask)
    hashes = get_kmer_hashes_numpy(numeric_reads, k=k)[mask]
    if include_reverse_complement:
        hashes_complement = get_kmer_hashes_numpy(numeric_reads_complement, is_complement=True, k=k)[mask]
    logging.info("Time to get kmer hashes: %.3f" % (time.time()-t))
    t = time.time()

    if include_reverse_complement:
        all_hashes = np.concatenate([hashes, hashes_complement])
    else:
        all_hashes = hashes  # np.concatenate([hashes, hashes_complement])

    logging.info("N hashes in total: %d" % len(all_hashes))
    logging.info("Time to concatenate hashes and complement hashes: %.3f" % (time.time()-t))
    t = time.time()

    if return_only_kmers:
        return all_hashes

    unique, counts = get_unique_kmers_and_counts(all_hashes)
    #logging.info("N unique hashes  : %d" % len(unique))
    logging.info("Time to count kmers: %.3f" % (time.time()-t))

    return unique, counts


def get_kmers_from_fasta(fasta_file_name, chunk_size=500000, k=31, max_read_length=150, return_only_kmers=False):
    t = time.time()
    #logging.info("Read %d reads" % reads.shape[0])
    #logging.info("Time reading from file: %.4f" % (time.time()-t))
    reads, mask = next(get_reads_as_matrices(fasta_file_name, chunk_size=chunk_size, max_read_length=max_read_length))
    return get_kmers_from_read_matrix(reads, mask, k=k, return_only_kmers=return_only_kmers)



def map_fasta_single_thread_with_numpy_parsing(data):
    logging.info("---------------------")
    logging.info("Starting processing chunk of reads")
    log_memory_usage_now()
    time_start = time.perf_counter()
    reads, args = data

    raw_chunk = from_shared_memory(args.buffer_type, reads)
    sequence_chunk = raw_chunk.get_sequences()
    logging.info("Size of sequence chunk (GB): %.3f" % (sequence_chunk.nbytes() / 1000000000))

    if args.use_numpy:
        kmer_counts = from_shared_memory(SingleSharedArray, "counts_shared"+args.random_id).array
    else:
        node_counts = from_shared_memory(SingleSharedArray, "counts_shared"+args.random_id).array

    t = time.perf_counter()
    hashes = TwoBitHash(k=args.kmer_size).get_kmer_hashes(sequence_chunk)
    logging.info("Size of hashes (GB): %.3f" % (hashes.nbytes / 1000000000))
    logging.info("Time spent to get %d kmer hashes: %.3f" % (len(hashes), time.perf_counter() - t))

    t = time.perf_counter()
    if args.use_numpy:
        kmer_index = from_shared_memory(NodeCount, args.kmer_index)
        kmer_counts += kmer_index.count_kmers(hashes)
    else:
        kmer_index = from_shared_memory(KmerIndex, args.kmer_index)
        node_counts += map_kmers_to_graph_index(kmer_index, args.n_nodes, hashes, args.max_hits_per_kmer)
    logging.info("Getting node counts took %.3f sec" % (time.perf_counter()-t))
    logging.info("Done with chunk. Took %.3f sec" % (time.perf_counter()-time_start))


def map_fasta(args, kmer_index):
    logging.info("Mapping fasta to kmer index %s" % args.kmer_index)
    args.random_id = str(np.random.random())
    n_nodes = kmer_index.max_node_id()
    logging.info("Putting kmer index in shared memory")
    args.kmer_index = to_shared_memory(kmer_index)

    args.n_nodes = n_nodes
    start_time = time.time()

    pool = get_shared_pool(args.n_threads)

    if args.use_numpy:
        n_kmers = kmer_index._kmers.size
        kmer_counts = np.zeros(n_kmers, dtype=np.uint32)
        to_shared_memory(SingleSharedArray(kmer_counts), "counts_shared" + args.random_id)
        kmer_counts = None
    else:
        node_counts = np.zeros(n_nodes+1, dtype=np.uint32)
        to_shared_memory(SingleSharedArray(node_counts), "counts_shared" + args.random_id)
        node_counts = None


    parser = BufferedNumpyParser.from_filename(args.fasta_file, args.chunk_size * 130)
    args.buffer_type = OneLineFastaBuffer2Bit
    chunks = parser.get_chunks()
    reads = (to_shared_memory(chunk) for chunk in chunks)
    func = map_fasta_single_thread_with_numpy_parsing

    i = 0
    data = zip(reads, repeat(args))

    for result in pool.imap(func, data, chunksize=1):
        logging.info("Done with %d chunks" % i)
        i += 1

    if args.use_numpy:
        kmer_counts = from_shared_memory(SingleSharedArray, "counts_shared"+args.random_id).array
        node_counts = kmer_index.get_node_counts_from_kmer_counts(kmer_counts)
    else:
        node_counts = from_shared_memory(SingleSharedArray, "counts_shared"+args.random_id).array            
    
    np.save(args.output_file, node_counts)
    logging.info("Saved node counts to %s.npy" % args.output_file)
    logging.info("Spent %.3f sec in total mapping kmers using %d threads" % (time.time()-start_time, args.n_threads))

    close_shared_pool()
