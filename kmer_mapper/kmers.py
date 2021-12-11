import numpy as np
from .parser import get_mask_from_intervals
from .encodings import twobit_swap
def get_kmer_mask(intervals, size, k=31):
    starts, ends = intervals
    ends = np.maximum(starts, ends-k+1)
    return get_mask_from_intervals((starts, ends), size-k+1)


class TwoBitHash:
    def __init__(self, k=31, dtype=np.uint64):
        self._dtype = dtype
        self.k = k
        self._mask = dtype(4**k-1)
        self._n_letters_in_dtype = 4*dtype(0).nbytes
        self._shifts = dtype(2)*np.arange(self._n_letters_in_dtype, dtype=dtype)
        self._rev_shifts = self._shifts[::-1]+dtype(2)

    def get_kmer_hashes(self, sequences):
        """Matching old interface"""
        last_end = sequences.intervals[1][-1]
        mask = get_kmer_mask(sequences.intervals, last_end, self.k)
        kmers = self.np_get_kmers(sequences.sequences.view(self._dtype))[:mask.size]
        reverse_hashes = sequences.encoding.complement(kmers) & self._dtype(4**self.k-1)
        forward_hashes = twobit_swap(kmers) >> ((self._n_letters_in_dtype-self.k)*2)
        return forward_hashes, reverse_hashes, mask

    def np_get_kmers_with_buffer(self, sequence):
        res = (sequence[:-1, None] >> self._shifts)
        res |= (sequence[1:, None] << self._rev_shifts)
        return res

    def np_get_kmers(self, sequence):
        assert sequence.dtype==self._dtype, sequence.dtype
        result = (sequence[:, None] >> self._shifts)
        result[:-1] |= (sequence[1:, None] << self._rev_shifts)
        result &= self._mask
        n_kmers = self._n_letters_in_dtype*sequence.size-self.k+1
        return result.ravel()[:n_kmers]

    def _has_buffer(self, sequences):
        last_end = sequences.intervals[1][-1]
        return sequences.sequences.size*4 - last_end>=self._n_letters_in_dtype-self.k+1

    def get_new_kmer_hashes(self, sequences):
        last_end = sequences.intervals[1][-1]
        mask = get_kmer_mask(sequences.intervals, last_end, self.k)
        if self._has_buffer(sequences):
            kmers = self.np_get_kmers_with_buffer(sequences.sequences.view(self._dtype)).ravel()
        else: 
            kmers = self.np_get_kmers(sequences.sequences.view(self._dtype))
        return kmers[:mask.size][mask] & self._mask

