import logging
import time
import numpy as np
from kmer_mapper.mapper import read_fasta_into_chunks
from kmer_mapper.util import remap_array, remap_array2
from scipy.ndimage import convolve1d
import pandas as pd
import scipy.signal

def get_reads_as_matrices(read_file_name, chunk_size=500000, max_read_length=150):
    return (chunk for chunk in read_fasta_into_chunks(read_file_name, chunk_size, max_read_length))


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
    return function(numeric_read_matrix.flatten(), power_array, mode="full")[k-1:].reshape(numeric_read_matrix.shape)[:,:-(k-1)]


def get_unique_kmers_and_counts(hash_matrix):
    return np.unique(hash_matrix, return_counts=True)
    unique = pd.value_counts(hash_matrix, sort=False)
    return unique.keys(), unique.values


def get_kmers_from_read_matrix(read_matrix, mask, k=31, return_only_kmers=False):
    logging.info("k=%d" % k)
    t = time.time()
    numeric_reads = convert_read_matrix_to_numeric(read_matrix)
    numeric_reads_complement = convert_read_matrix_to_numeric(read_matrix, True)
    logging.info("Time to convert reads to numeric: %.3f" % (time.time()-t))
    t = time.time()

    mask = mask[:,k-1:]  # make mask for hashes (we don't want hashes for bases not from reads, which are False in original mask)
    hashes = get_kmer_hashes_numpy(numeric_reads, k=k)[mask]
    hashes_complement = get_kmer_hashes_numpy(numeric_reads_complement, is_complement=True, k=k)[mask]
    logging.info("Time to get kmer hashes: %.3f" % (time.time()-t))
    t = time.time()

    all_hashes = np.concatenate([hashes, hashes_complement])
    logging.info("N hashes in total: %d" % len(all_hashes))
    logging.info("Time to concatennate hashes and complement hashes: %.3f" % (time.time()-t))
    t = time.time()

    if return_only_kmers:
        return all_hashes

    unique, counts = get_unique_kmers_and_counts(all_hashes)
    logging.info("N unique hashes  : %d" % len(unique))
    logging.info("Time to count kmers: %.3f" % (time.time()-t))
    t = time.time()

    return unique, counts


def get_kmers_from_fasta(fasta_file_name, chunk_size=500000, k=31, max_read_length=150, return_only_kmers=False):
    t = time.time()
    reads, mask = next(get_reads_as_matrices(fasta_file_name, chunk_size=chunk_size, max_read_length=max_read_length))
    logging.info("Read %d reads" % reads.shape[0])
    logging.info("Time reading from file: %.4f" % (time.time()-t))
    return get_kmers_from_read_matrix(reads, mask, k=k, return_only_kmers=return_only_kmers)




