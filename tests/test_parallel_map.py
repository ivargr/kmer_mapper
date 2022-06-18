import logging
logging.basicConfig(level=logging.INFO)
import time

from kmer_mapper.mapping import ParalellMapper
from graph_kmer_index import KmerIndex
from shared_memory_wrapper import get_shared_pool


def test():
    n_threads = 4
    get_shared_pool(n_threads)
    index = KmerIndex.from_file("kmer_index_only_variants.npz")
    index.remove_ref_offsets()
    index.convert_to_int32()

    max_node_id = 10000000
    t = time.perf_counter()
    mapper = ParalellMapper(index, max_node_id, n_threads=n_threads)
    kmers = index._kmers
    mapper.map(kmers)
    results = mapper.get_results()

    logging.info("time spent: %.3f" % (time.perf_counter()-t))

    print(results)


if __name__ == "__main__":
    test()
