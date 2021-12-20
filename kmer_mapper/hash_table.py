import numpy as np

class ModuloHashLookup:
    def __init__(self, values, mod, n_entries, lookup_end):
        self._values = values # np.append(sorted_values, np.array([4**31], dtype=np.uint64))
        self._mod = mod
        self._n_entries = n_entries
        self._lookup_end = lookup_end

    def _get_hash(self, queries):
        return queries % self._mod

    def find_matches(self, queries):
        hashes = self._get_hash(queries)
        n_entries = self._n_entries[hashes]
        hit_idxs = np.flatnonzero(n_entries)
        n_entries = n_entries[hit_idxs]
        n_iters = np.floor(np.log2(n_entries)).astype(int)+1 #Required iterations for binary search
        args = np.argsort(n_iters)
        hit_idxs = hit_idxs[args]
        R = self._lookup_end[hashes[hit_idxs]]
        L = R - n_entries[args]
        queries = queries[hit_idxs]
        found_indices = self._binary_search(queries, L, R, n_iters)
        return found_indices[self._values[found_indices]==queries]

    def get_index(self, kmers):
        row_numbers = self._get_hash(kmers)
        n_entries = self._n_entries[row_numbers]
        assert np.min(n_entries)>0, np.min(n_entries)
        R = self._lookup_end[row_numbers]
        L = R - n_entries
        n_iters = np.floor(np.log2(n_entries)).astype(int)+1
        n_iters = np.maximum(n_iters, np.max(n_iters))
        found_indices = self._binary_search(kmers, L, R, n_iters)
        return found_indices

    def _binary_search(self, queries, L, R, n_iters):
        """
        https://en.wikipedia.org/wiki/Binary_search_algorithm#Alternative_procedure
        Look for queries in range [L, R>
        Runs only reuqired iterations for each group
        """
        boundries = np.vstack((L, R))
        indexes = np.arange(boundries.shape[-1])
        stops = np.cumsum(np.bincount(n_iters))
        for stop in stops[:-1]:
            m = (boundries[:, stop:].sum(axis=0))//2
            is_larger = (self._values[m] > queries[stop:]).view(np.uint8)
            boundries[is_larger, indexes[stop:]] = m
        return boundries[0]

    @classmethod
    def from_values_and_mod(cls, keys, modulo):
        hashes = (keys % modulo).astype(int)
        counts = np.bincount(hashes, minlength=modulo)
        args = np.lexsort([keys, hashes])
        hash_sorted = keys[args]
        n_entries=counts
        lookup_end = np.cumsum(counts)
        return cls(hash_sorted, modulo, n_entries, lookup_end)

class NodeCount:
    dtype=np.uint64
    k=31
    lookup_class = ModuloHashLookup
    n_bins = 200000003
    def __init__(self, kmers, kmer_indexes, node_ids):
        self._kmers = kmers
        self._node_ids = node_ids
        self._kmer_indexes = kmer_indexes

    def index_kmers(self):
        self._indexed_lookup = self.lookup_class.from_values_and_mod(
            self._kmers, np.uint64(self.n_bins))
        self._kmer_indexes = self._indexed_lookup.get_index(self._kmer_indexes)

    def get_node_counts(self, kmers):
        counts = self.count_kmers(kmers)
        return np.bincount(self._node_ids, counts[self._kmer_indexes])

    def count_kmers(self, kmers):
        indices = self._indexed_lookup.find_matches(kmers)
        return np.bincount(indices, minlength=self._kmers.size)

    @classmethod
    def from_old_file(cls, filename):
        D = np.load(filename)
        kmers = D["kmers"]
        repr_kmers = kmers[D["representative_kmers"]]
        obj = cls(kmers, repr_kmers, D["lookup"])
        obj.index_kmers()
        return obj

    @classmethod
    def from_file(cls, filename):
        D = np.load(filename)
        obj = cls(D["kmers"], D["kmer_indexes"], D["node_ids"])
        obj._indexed_lookup=cls.lookup_class(
            D["kmers"], D["mod"], D["n_entries"], D["lookup_end"])
        return obj

    @classmethod
    def from_old_index_files(cls, filename):
        logging.info("From old index files")
        data = np.load(filename)
        kmers = data["kmers"]
        unique_kmers = np.unique(kmers)
        nodes = data["nodes"]
        k = cls(unique_kmers, kmers, nodes)
        k.index_kmers()
        return k

    def to_file(self, filename):
        np.savez(filename, 
                 kmers=self._indexed_lookup._values,
                 kmer_indexes=self._kmer_indexes,
                 node_ids=self._node_ids,
                 lookup_end=self._indexed_lookup._lookup_end,
                 mod=self._indexed_lookup._mod,
                 n_entries=self._indexed_lookup._n_entries)
