import numpy as np
import logging


class GpuCounter:
    def __init__(self, unique_kmers, kmers, nodes):
        self.unique_kmers = unique_kmers
        self.kmers = kmers
        self.nodes = nodes

    def initialize_cuda(self, modulo):
        from cucounter.counter import Counter
        logging.info("N unique kmers: %d" % len(self.unique_kmers))
        self.counter = Counter(self.unique_kmers, modulo)

    @classmethod
    def from_kmers_and_nodes(cls, kmers, nodes):
        unique_kmers = np.unique(kmers)
        return cls(unique_kmers, kmers, nodes)

    def count(self, kmers):
        self.counter.count(kmers)

    def get_node_counts(self, min_nodes=0):
        return np.bincount(self.nodes, self.counter[self.kmers], minlength=min_nodes)

