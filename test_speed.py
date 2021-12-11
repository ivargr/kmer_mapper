import time
from graph_kmer_index import KmerIndex
from mapper import map_kmers_to_graph_index, map_kmers_to_graph_index_test, map_kmers_to_graph_index_test2
from shared_memory_wrapper import to_shared_memory, from_shared_memory

def test(index, max_node_id, kmers):
    t = time.perf_counter()
    #nodes = map_kmers_to_graph_index_test(index, max_node_id, kmers)
    nodes = map_kmers_to_graph_index_test(index, max_node_id, kmers)
    print(time.perf_counter()-t)






index = KmerIndex.from_file("tests/kmer_index_only_variants.npz")
to_shared_memory(index, "test")

index2 = from_shared_memory(KmerIndex, "test")

#index2._modulo = index._modulo
#index2._hashes_to_index = index2._hashes_to_index.copy()
#index2._nodes = index._nodes
#index2._kmers = index._kmers
#index2._n_kmers = index._n_kmers

index3 = index2.copy()

max_node_id = index.max_node_id()
kmers = index._kmers[0:1000000]
#print(len(kmers), " kmers")

test(index, max_node_id, kmers)

test(index2, max_node_id, kmers)

test(index3, max_node_id, kmers)

test(index2, max_node_id, kmers)
