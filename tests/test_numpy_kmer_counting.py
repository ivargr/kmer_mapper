from kmer_mapper.kmer_counting import SimpleKmerLookup
from graph_kmer_index import KmerIndex


lookup = SimpleKmerLookup.from_old_index_files("kmer_index_only_variants.npz")
result = lookup.get_node_counts([1, 2, 3, 4])

print(result)



