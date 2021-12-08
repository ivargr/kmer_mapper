from kmer_mapper.kmer_lookup import Advanced2
from graph_kmer_index import KmerIndex
import numpy as np
np.random.seed(1)

def test():
    old_index = KmerIndex.from_file("kmer_index_only_variants.npz")
    new_index = Advanced2.from_old_index_files("kmer_index_only_variants.npz")
    kmers = old_index._kmers

    for kmer in kmers:
        old_hits = old_index.get(int(kmer), max_hits=1000000000)[0]
        kmer = np.array([kmer], dtype=np.uint64)
        new_hits = np.nonzero(new_index.get_node_counts(kmer))[0]

        print(kmer, old_hits, new_hits)

        assert set(old_hits) == set(new_hits), "Fail for kmer %d" % kmer


test()