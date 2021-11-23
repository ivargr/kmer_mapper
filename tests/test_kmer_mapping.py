import time
import numpy as np
from kmer_mapper.mapper import read_fasta_into_chunks
from kmer_mapper.util import remap_array, remap_array2
from scipy.ndimage import convolve1d

def get_reads_as_matrices(read_file_name, chunk_size=500000):
    return (chunk for chunk in read_fasta_into_chunks(read_file_name, chunk_size))


def convert_read_matrix_to_numeric(read_matrix, give_complement_base_values=False):
    # from byte values
    from_values = np.array([65, 67, 71, 84, 97, 99, 103, 116])  # NB: Must be increasing
    # to internal base values for a, c, t, g
    to_values = np.array([0, 1, 3, 2, 0, 1, 3, 2])
    if give_complement_base_values:
        to_values = np.array([2, 3, 1, 0, 2, 3, 1, 0])

    return remap_array(read_matrix, from_values, to_values)


def get_kmer_hashes(numeric_read_matrix, is_complement=False, k=31):
    power_array = np.power(4, np.arange(0, k))
    if is_complement:
        power_array = np.power(4, np.arange(0, k)[::-1])

    k_half = k//2  # we don't want first k half and last k half columns
    return convolve1d(numeric_read_matrix, power_array, mode="constant")[:,k_half:-k_half]


def get_unique_kmers_and_counts(hash_matrix):
    return np.unique(hash_matrix, return_counts=True)

k = 31

start = time.time()
read_matrix, mask = next(get_reads_as_matrices("tests/hg002_simulated_reads_15x.fa"))
print(mask)
print("Read reads into matrix: %.4f" % (time.time()-start))
start = time.time()

numeric_reads = convert_read_matrix_to_numeric(read_matrix)
numeric_reads_complement = convert_read_matrix_to_numeric(read_matrix, True)
print("Remapping to base values: %.4f" % (time.time()-start))
start = time.time()

hashes = get_kmer_hashes(numeric_reads)[:,k-1:]
hashes_complement = get_kmer_hashes(numeric_reads)[:,k-1:]
print("Getting hashes: %.4f" % (time.time()-start))
start = time.time()

unique, counts = get_unique_kmers_and_counts(hashes)
unique_complement, counts_complement = get_unique_kmers_and_counts(hashes_complement)
print("Getting unique kmers and counts of each: %.4f" % (time.time()-start))

print("N unique hashes: %d" % len(unique))
print("N total hashes: %d" % len(hashes))



