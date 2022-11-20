import numpy as np
import logging


class GpuCounter:
    def __init__(self, unique_kmers, kmers, nodes, k):
        self.unique_kmers = unique_kmers
        self.kmers = kmers
        self.nodes = nodes
        self.counter = None
        self.k = k

    def initialize_cuda(self, modulo):
        from cucounter.counter import Counter
        print("N unique kmers: %d" % len(self.unique_kmers))
        self.counter = Counter(self.unique_kmers, modulo)

    @classmethod
    def from_kmers_and_nodes(cls, kmers, nodes, k) -> "GpuCounter":
        unique_kmers = np.unique(kmers)
        return cls(unique_kmers, kmers, nodes, k)

    def count(self, kmers, count_revcomps=False):
        self.counter.count(kmers, count_revcomps, self.k)

    def get_node_counts(self, min_nodes=0):
        # get counts in chunks to lower memory
        counts = np.zeros(len(self.kmers), dtype=np.uint32)
        chunk_size = 10_000_000
        start = 0
        for chunk in np.array_split(self.kmers, len(self.kmers)//chunk_size):
            print("Querying chunk %d-%d" % (start, start+len(chunk)))
            counts[start:start+len(chunk)] = self.counter[chunk]
            start += len(chunk)

        logging.info("Doing bincount")
        return np.bincount(self.nodes, counts, minlength=min_nodes)

