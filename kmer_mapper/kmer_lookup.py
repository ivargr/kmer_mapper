import numpy as np


class KmerLookup:
    def __init__(self, kmers, representative_kmers, lookup):
        self._kmers = kmers
        self._representative_kmers = representative_kmers
        self._lookup = lookup

    def index_kmers(self):
        self._kmers.sort()
        self._representative_kmers = np.searchsorted(self._kmers, self._representative_kmers)

    def _get_indexes(self, kmers):
        return np.searchsorted(self._kmers, kmers)

    def count_kmers(self, kmers):
        indexes = self._get_indexes(kmers)
        return np.bincount(indexes[kmers==self._kmers[indexes]], minlength=self._kmers.size)
        
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
        return cls(unique_kmers, kmers, data["nodes"])
