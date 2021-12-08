import numpy as np


class KmerLookup:
    def __init__(self, kmers, representative_kmers, lookup):
        self._kmers = kmers
        self._representative_kmers = representative_kmers
        self._lookup = lookup

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
        data = np.load(filename)
        kmers = data["kmers"]
        unique_kmers = np.unique(kmers)
        k = cls(unique_kmers, kmers, data["nodes"])
        k.index_kmers()
        return k

class AdvancedKmerLookup(KmerLookup):
    n_bins = 20000000
    max_value = 2**31
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

class IndexedSortedLookup(SimpleKmerLookup):
    def __init__(self, sorted_values, index, mod, row_sizes):
        self._sorted_values = np.append(sorted_values, 4**31)
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
        print(values, index, row_sizes)
        return cls(values, index[:-1], mod, row_sizes)

    def find_matches(self, queries):
        row_numbers = queries // self._mod
        row_sizes = self._row_sizes[row_numbers]
        mask = row_sizes > 0
        row_sizes = row_sizes[mask]
        L = self._index[row_numbers[mask]]
        R = L + row_sizes-1
        # boundries = self._boundries[row_numbers[mask]].copy()
        n_iter = int(np.log2(np.max(row_sizes)))+1
        queries = queries[mask]
        found_indices = self._binary_search(queries, L, R, n_iter)
        return L[self._sorted_values[found_indices]==queries]

    def _binary_search(self, queries, L, R, n_iter):
        """
        https://en.wikipedia.org/wiki/Binary_search_algorithm#Alternative_procedure
        """
        boundries = np.vstack((L, R))
        indexes = np.arange(boundries.shape[-1])
        for _ in range(n_iter):
            m = (boundries.sum(axis=0)+1)//2
            is_larger = (self._sorted_values[m] > queries).astype(int)
            boundries[is_larger, indexes]= m-is_larger
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
        print(n_iter)
        for _ in range(n_iter):
            m = (L+R+1)//2
            is_larger = self._sorted_values[m] > queries
            R[is_larger] = m[is_larger]-1
            L[~is_larger] = m[~is_larger]
        return L

class HashedKmerLookup(SimpleKmerLookup):
    def count_kmers(self, kmers):
        hashes = kmers % self.mod
        single_hit_mask = self._single_hit_mask[hashes]
        multi_hit_mask =  self._multi_hit_mask[hashes]
        single_hit_idxs = self._hash_to_kmer_idx[single_hit_mask]
        multi_hit_idxs = self._get_indexes(kmers[multi_hit_mask])
        all_idxs = np.concatenate([
            single_hit_idxs[self._kmers[single_hit_idxs]==kmers[single_hit_mask]],
            multi_hit_idxs[self._kmers[multi_hit_idxs]==kmers[multi_hit_mask]]
            ])
        return np.bincount(all_idxs, minlength=self._kmers.size)


        hash_hit_mask = self._hash_to_kmer[hashes] == kmers
        kmers = kmers[n_hits==1] == self._hash_to_kmer[hashes[n_hist==1]]

    def _create_hash_table(self, sorted_kmers):
        hashes = kmers % self.mod
        n_hits = np.bincount(hashes, minlength=mod)
        self._single_hit_mask = n_hits==1
        self._multi_hit_mask = n_hits>1
        self._hash_to_kmer_idx = np.empty_like(n_hits)
        self._hash_to_kmer_idx[hashes] = np.arange(kmers.size)
        
