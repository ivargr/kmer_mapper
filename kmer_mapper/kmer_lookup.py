import logging
from .encodings import twobit_swap, ACTGTwoBitEncoding
import numpy as np


class KmerLookup:
    def __init__(self, kmers, representative_kmers, lookup):
        self._kmers = kmers
        self._representative_kmers = representative_kmers
        self._lookup = lookup

    def max_node_id(self):
        return np.max(self._lookup)

    def to_file(self, file_name):
        np.savez(file_name, 
                 kmers = self._kmers,
                 representative_kmers = self._representative_kmers,
                 lookup = self._lookup)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name)
        return cls(data["kmers"],
                   data["representative_kmers"],
                   data["lookup"])



    def index_kmers(self):
        self._kmers.sort()
        self._representative_kmers = np.searchsorted(self._kmers, self._representative_kmers)

    def _get_indexes(self, kmers):
        return np.searchsorted(self._kmers, kmers)

    def count_kmers(self, kmers):
        indexes = self._get_indexes(kmers)
        return np.bincount(indexes[kmers==self._kmers[np.minimum(indexes, self._kmers.size-1)]],
                           minlength=self._kmers.size)
        
    def get_node_counts(self, kmers):
        counts = self.count_kmers(kmers)
        cum_counts = np.insert(np.cumsum(counts[self._representative_kmers]), 0, 0)
        node_counts = cum_counts[self._lookup[:, 1]]-cum_counts[self._lookup[:, 0]]
        return node_counts

class SimpleKmerLookup(KmerLookup):
    def get_node_counts(self, kmers):
        counts = self.count_kmers(kmers)
        return np.bincount(self._lookup, counts[self._representative_kmers])

    @classmethod
    def from_old_index_files(cls, filename):
        logging.info("From old index files")
        data = np.load(filename)
        kmers = data["kmers"]
        unique_kmers = np.unique(kmers)
        k = cls(unique_kmers, kmers, data["nodes"])
        k.index_kmers()
        return k

class AdvancedKmerLookup(SimpleKmerLookup):
    n_bins = 200000000
    max_value = 4**31
    def _get_indexes(self, kmers):
        return self._indexed_lookup.find_queries(kmers)

    def index_kmers(self):
        self._kmers.sort()
        self._representative_kmers = np.searchsorted(self._kmers, self._representative_kmers)
        self._indexed_lookup = IndexedSortedLookup.from_values_and_mod(
            self._kmers, self.max_value//self.n_bins+1,
            self.max_value, do_sort=False)

class Advanced2(AdvancedKmerLookup):
    def count_kmers(self, kmers):
        indices = self._indexed_lookup.find_matches(kmers)
        return np.bincount(indices, minlength=self._kmers.size)

    @classmethod
    def from_file(cls, filename):
        D = np.load(filename)
        obj = cls(D["kmers"], D["representative_kmers"], D["lookup"])
        obj._indexed_lookup=IndexedSortedLookup(
            D["kmers"], D["index"], D["mod"], D["row_size"])
        return obj

    def to_file(self, filename):
        np.savez(filename, 
                 kmers=self._kmers,
                 representative_kmers=self._representative_kmers,
                 lookup=self._lookup,
                 index=self._indexed_lookup._index,
                 mod=self._indexed_lookup._mod,
                 row_size=self._indexed_lookup._row_sizes)
    

class NewHashLookup(Advanced2):
    dtype=np.uint64
    k=31
    @classmethod
    def _rehash_kmers(cls, kmers):
        new_hashes = twobit_swap(kmers)>>cls.dtype(2)
        reverse_hashes = ACTGTwoBitEncoding.complement(kmers) & cls.dtype(4**cls.k-1)
        return np.concatenate((new_hashes, reverse_hashes))

    @classmethod
    def from_lookup_with_old_hash(cls, old_lookup):
        kmers = cls._rehash_kmers(old_lookup._representative_kmers)
        nodes = np.concatenate((old_lookup._lookup, old_lookup._lookup))
        ret = cls(np.unique(kmers), kmers, nodes)
        ret.index_kmers()
        return ret

    @classmethod
    def from_old_index_files(cls, filename):
        logging.info("From old index files")
        data = np.load(filename)
        kmers = cls._rehash_kmers(data["kmers"])
        unique_kmers = np.unique(kmers)
        nodes = data["nodes"]
        nodes = np.concatenate((nodes, nodes))
        k = cls(unique_kmers, kmers, nodes)
        k.index_kmers()
        return k

class IndexedSortedLookup(SimpleKmerLookup):
    def __init__(self, sorted_values, index, mod, row_sizes):
        self._sorted_values = np.append(sorted_values, np.array([4**31], dtype=np.uint64))
        self._mod = mod
        self._index = index
        self._row_sizes = row_sizes
        # THis is maybe to memory demanding
        # self._boundries = np.vstack((self._index, self._index+self._row_sizes-1))

    @classmethod
    def from_values_and_mod(cls, values, mod, max_value, do_sort=False):
        if do_sort:
            values = np.sort(values)
        m = values//mod
        index = np.searchsorted(m, np.arange(max_value//mod+2))
        row_sizes = index[1:]-index[:-1]
        return cls(values, index[:-1], mod, row_sizes)

    def _get_matched_data(self, row_numbers):
        row_sizes = self._row_sizes[row_numbers]
        mask = row_sizes > 0
        row_sizes = row_sizes[mask]
        L = self._index[row_numbers[mask]]
        R = L + row_sizes
        return L, R, mask, np.max(row_sizes)


    def find_matches(self, queries):
        L, R, mask, row_length = self._get_matched_data(queries // self._mod)
        n_iter = int(np.log2(row_length))+1
        queries = queries[mask]
        found_indices = self._binary_search(queries, L, R, n_iter)
        return found_indices[self._sorted_values[found_indices]==queries]

    def _binary_search(self, queries, L, R, n_iter):
        """
        https://en.wikipedia.org/wiki/Binary_search_algorithm#Alternative_procedure
        """
        boundries = np.vstack((L, R))
        indexes = np.arange(boundries.shape[-1])
        for _ in range(n_iter):
            m = (boundries.sum(axis=0))//2
            is_larger = (self._sorted_values[m] > queries).view(np.uint8)
            boundries[is_larger, indexes] = m
            """
            m = (L+R+1)//2
            is_larger = self._sorted_values[m] > queries
            R[is_larger] = m[is_larger]-1
            L[~is_larger] = m[~is_larger]
            """
        return boundries[0] 
                          
    def find_queries(self, queries):
        row_numbers = queries // self._mod
        L = self._index[row_numbers]
        row_sizes = self._row_sizes[row_numbers]
        R = L + row_sizes-1
        m = L
        n_iter = int(np.log2(np.max(row_sizes)))+1
        for _ in range(n_iter):
            m = (L+R+1)//2
            is_larger = self._sorted_values[m] > queries
            R[is_larger] = m[is_larger]-1
            L[~is_larger] = m[~is_larger]
        return L

        
