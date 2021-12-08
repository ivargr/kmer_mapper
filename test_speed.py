import time
from graph_kmer_index import KmerIndex
from kmer_mapper.mapping import map_kmers_to_graph_index, map_kmers_to_graph_index_test
from shared_memory_wrapper import to_shared_memory, from_shared_memory

def test(index, max_node_id, kmers):
    t = time.perf_counter()
    nodes = map_kmers_to_graph_index(index, max_node_id, kmers)
    print(time.perf_counter()-t)






index = KmerIndex.from_file("kmer_index_only_variants.npz")
to_shared_memory(index, "test")

index2 = from_shared_memory(KmerIndex, "test")
"""
index2._modulo = index._modulo
index2._hashes_to_index = index._hashes_to_index
index2._nodes = index._nodes
index2._kmers = index._kmers
index2._n_kmers = index._n_kmers
"""

index3 = index2.copy()

print(type(index._kmers))
print(type(index2._kmers))
print(index._modulo, index2._modulo)
print(type(index._modulo))
print(type(index2._modulo))
max_node_id = index.max_node_id()
kmers = index._kmers[0:1000000]

test(index, max_node_id, kmers)


print(index)
print(index2)
test(index2, max_node_id, kmers)

test(index3, max_node_id, kmers)
