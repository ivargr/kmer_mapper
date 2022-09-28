import logging
import time
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

import pyximport
pyximport.install()
import kmer_mapper.mapper

from graph_kmer_index.collision_free_kmer_index import MinimalKmerIndex, CounterKmerIndex
import numpy as np
import argparse
from graph_kmer_index import KmerIndex
from .mapping import map_fasta
from graph_kmer_index.index_bundle import IndexBundle
from .kmer_lookup import Advanced2
from shared_memory_wrapper.shared_memory import remove_shared_memory_in_session
from shared_memory_wrapper.shared_memory import get_shared_pool, close_shared_pool
from shared_memory_wrapper import from_file, object_to_shared_memory, object_from_shared_memory
import bionumpy as bnp
from kmer_mapper.mapper import map_kmers_to_graph_index
from bionumpy.kmers import fast_hash
from bionumpy.encodings import ACTGTwoBitEncoding


def main():
    run_argument_parser(sys.argv[1:])


def _get_kmer_index_from_args(args):
    if args.kmer_index is None:
        if args.index_bundle is None:
            logging.error("Either a kmer index (-i) or an index bundle (-b) needs to be specified")
            sys.exit(1)
        else:
            kmer_index = IndexBundle.from_file(args.index_bundle).indexes["KmerIndex"]
            kmer_index.convert_to_int32()
            kmer_index.remove_ref_offsets()  # not needed, will save us some memory
    else:

        cls = KmerIndex
        if "minimal" in args.kmer_index:
            cls = MinimalKmerIndex
        kmer_index = cls.from_file(args.kmer_index)
        kmer_index.convert_to_int32()
        kmer_index.remove_ref_offsets()  # not needed, will save us some memory

    return kmer_index


def map_fasta_command(args):
    if args.kmer_size > 31:
        logging.error("k must be 31 or lower")
        sys.exit(1)

    logging.info("Max read length is specified to %d" % args.max_read_length)
    logging.info("Max hits per kmer: %d" % args.max_hits_per_kmer)

    if not args.fasta_file.endswith(".fa"):
        logging.error("Only fasta files (not fq or gzipped files) are supported to the argument -f. If you have another"
                          "format, you can pipe fasta to kmer_mapper and use a dash as file name (-f -)")
        sys.exit(1)

    get_shared_pool(args.n_threads)
    kmer_index = None

    kmer_index = _get_kmer_index_from_args(args)

    args.max_node_id = kmer_index.max_node_id()
    logging.info("Max node id is %d" % args.max_node_id)
    map_fasta(args, kmer_index)


def map_bnp(args):
    get_shared_pool(args.n_threads)
    k = args.kmer_size

    kmer_index = _get_kmer_index_from_args(args)

    hash_time = 0
    t_start = time.perf_counter()

    chunks = bnp.open(args.reads, chunk_size=args.chunk_size)
    n_processed = 0
    for chunk in chunks:
        #print(type(chunk.sequence.dtype))
        #print(chunk.to_sequences())
        t0 = time.perf_counter()
        #hashes = chunk.sequence
        hashes = fast_hash(chunk.sequence, args.kmer_size).ravel().astype(np.uint64)
        hashes = ACTGTwoBitEncoding.complement(hashes) & np.uint64(4**k-1)
        hashes_copy = np.zeros_like(hashes)

        hash_time += time.perf_counter()-t0
        #hashes[0] = 11
        print(hashes[::-1])
        print(hashes)


        mapped = map_kmers_to_graph_index(kmer_index, kmer_index.max_node_id(), hashes_copy)
        print(mapped)
        print("Mapped: %d" % np.sum(mapped))


        id = object_to_shared_memory(chunk.sequence)
        sequence2 = object_from_shared_memory(id)

        n_processed += len(chunk.sequence)
        logging.info("%d reads processed. Hash time: %.3f" % (n_processed, hash_time))

        print(type(chunk.sequence.ravel()))

    logging.info("Hash time: %.3f" % (hash_time))
    logging.info("Total time: %.3f" % (time.perf_counter()-t_start))
    close_shared_pool()
    return

    """
    mapper = ParalellMapper(kmer_index, kmer_index.max_node_id(), args.n_threads)
    t = time.perf_counter()
    mapper.map_reads_file(args.reads, args.kmer_size, args.chunk_size)
    node_counts = mapper.get_results()
    logging.info("Spent %.3f sec to map" % (time.perf_counter()-t))
    close_shared_pool()
    np.save(args.output_file, node_counts)
    """




def run_argument_parser(args):

    parser = argparse.ArgumentParser(
        description='Kmer Mapper',
        prog='kmer_mapper',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("map")
    subparser.add_argument("-i", "--kmer-index", required=False)
    subparser.add_argument("-b", "--index-bundle", required=False)
    subparser.add_argument("-f", "--fasta-file", required=True)
    subparser.add_argument("-k", "--kmer-size", required=False, default=31, type=int)
    subparser.add_argument("-t", "--n-threads", required=False, default=16, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, type=int, default=500000, help="N reads to process in each chunk")
    subparser.add_argument("-l", "--max-read-length", required=False, type=int, default=150,
                           help="Maximum length of reads. Reads should not be longer than this.")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-n", "--use-numpy", required=False, type=bool, default=False, help="Use numpy-based counting instead of Cython")
    #subparser.add_argument("-N", "--use-cython-file-reading", required=False, type=bool, default=False, help="Use cython-based file reading instead of Cython")
    #subparser.add_argument("-m", "--no-shared-memory", required=False, type=bool, default=False, help="Set to True to not use shared memory for index. Increases memory usage by a factor of --n-threads.")
    #subparser.add_argument("-r", "--include-reverse-complement", required=False, default=False, type=bool)
    #subparser.add_argument("-T", "--use-numpy-parsing", required=False, default=False, type=bool)
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_fasta_command)



    subparser = subparsers.add_parser("map_bnp")
    subparser.add_argument("-i", "--kmer-index", required=False)
    subparser.add_argument("-b", "--index-bundle", required=False)
    subparser.add_argument("-f", "--reads", required=True, help="Reads in .fa, .fq, .fa.gz, or fq.gz format")
    subparser.add_argument("-k", "--kmer-size", required=False, default=31, type=int)
    subparser.add_argument("-t", "--n-threads", required=False, default=16, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, type=int, default=5000000,
                           help="N bytes to process in each chunk")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_bnp)




    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

    remove_shared_memory_in_session()
