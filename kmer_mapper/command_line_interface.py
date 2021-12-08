import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
import numpy as np
import argparse
from graph_kmer_index import KmerIndex
from .mapping import get_kmers_from_fasta, map_fasta
from mapper import map_kmers_to_graph_index
import time
from graph_kmer_index.index_bundle import IndexBundle
from .kmer_counting import SimpleKmerLookup
from .kmer_lookup import Advanced2
from .parser import OneLineFastaParser, KmerHash, Sequences
from shared_memory_wrapper import from_shared_memory, to_shared_memory, SingleSharedArray
from shared_memory_wrapper.shared_memory import remove_shared_memory_in_session
from pathos.multiprocessing import Pool
from itertools import repeat

def main():
    run_argument_parser(sys.argv[1:])

def test(args):
    print("Hi")


def map_using_numpy_single_thread(args):
    kmer_index = from_shared_memory(KmerIndex, "kmer_index"+args.shared_memory_id)

def map_using_numpy(data):
    reads, args = data
    logging.info("Reading from shared memory")
    kmer_index = from_shared_memory(KmerIndex, "kmer_index"+args.random_id)
    sequence_chunk = from_shared_memory(Sequences, reads)
    node_counts = from_shared_memory(SingleSharedArray, "counts_shared"+args.random_id).array
    logging.info("Done reading from shared memory")
    t = time.perf_counter()
    hashes, reverse_complement_hashes, mask = KmerHash(k=31).get_kmer_hashes(sequence_chunk)
    logging.info("time to get kmer hashes: %.3f" % (time.perf_counter()-t))
    for h in (hashes, reverse_complement_hashes):
        h = h[mask]
        t = time.perf_counter()
        node_counts += map_kmers_to_graph_index(kmer_index, args.n_nodes, h, args.max_hits_per_kmer)
        logging.info("Done mapping to kmer index. Took %.3f sec" % (time.perf_counter()-t))

    logging.info("Done with chunk")


def map_using_numpy_parallel(args):
    start_time = time.perf_counter()

    n_nodes = args.kmer_index.max_node_id()
    args.n_nodes = n_nodes
    node_counts = np.zeros(n_nodes + 1, dtype=float)
    if args.n_threads == 1:
        to_shared_memory(args.kmer_index, "kmer_index")
        args.kmer_index = from_shared_memory(KmerIndex, "kmer_index")
        fasta_parser = OneLineFastaParser(args.fasta_file, args.chunk_size * 150 // 3)
        reads = fasta_parser.parse(as_shared_memory_object=False)
        for sequence_chunk in reads:
            t = time.perf_counter()
            hashes, reverse_complement_hashes, mask = KmerHash(k=31).get_kmer_hashes(sequence_chunk)
            logging.info("time to get kmer hashes: %.3f" % (time.perf_counter() - t))
            for h in (hashes, reverse_complement_hashes):
                h = h[mask]
                t = time.perf_counter()
                node_counts += map_kmers_to_graph_index(args.kmer_index, args.n_nodes, h, args.max_hits_per_kmer)
                logging.info("Done mapping to kmer index. Took %.3f sec" % (time.perf_counter() - t))
    else:
        args.random_id = str(np.random.random())
        logging.info("Writing to shared memory")
        to_shared_memory(args.kmer_index, "kmer_index" + args.random_id)
        to_shared_memory(SingleSharedArray(node_counts), "counts_shared" + args.random_id)
        logging.info("Done writing to shared memory")

        args.kmer_index = None

        pool = Pool(args.n_threads)
        fasta_parser = OneLineFastaParser(args.fasta_file, args.chunk_size*150 // 3)
        reads = fasta_parser.parse(as_shared_memory_object=True)

        logging.info("Got reads, starting processes")
        for result in pool.imap(map_using_numpy, zip(reads, repeat(args))):
            continue

        node_counts = from_shared_memory(SingleSharedArray, "counts_shared" + args.random_id).array

    end_time = time.perf_counter()
    logging.info("Mapping all kmers took %.3f sec" % (end_time - start_time))
    np.save(args.output_file, node_counts)


def map_fasta_command(args):
    if args.kmer_size > 31:
        logging.error("k must be 31 or lower")
        sys.exit(1)

    if not args.fasta_file.endswith(".fa"):
        logging.error("Only fasta files (not fq or gzipped files) are supported for now.")
        sys.exit(1)


    if args.kmer_index is None:
        if args.index_bundle is None:
            logging.error("Either a kmer index (-i) or an index bundle (-b) needs to be specified")
            sys.exit(1)
        else:
            index = IndexBundle.from_file(args.index_bundle).indexes
            args.kmer_index = index["KmerIndex"]
    else:
        if args.use_numpy:
            logging.info("Using numpy index")
            args.kmer_index = SimpleKmerLookup.from_old_index_files(args.kmer_index)
        else:
            args.kmer_index = KmerIndex.from_file(args.kmer_index)

    if args.use_numpy_file_reading:
        map_using_numpy_parallel(args)
    else:
        map_fasta(args)


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
    subparser.add_argument("-n", "--use-numpy", required=False, type=bool, default=False, help="Use numpy-based counting instead of Cython")
    subparser.add_argument("-N", "--use-numpy-file-reading", required=False, type=bool, default=False, help="Use numpy-based file reading instead of Cython")
    subparser.add_argument("-l", "--max-read-length", required=False, type=int, default=150,
                           help="Maximum length of reads. Reads should not be longer than this.")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-r", "--ignore-reverse-complement", required=False, default=False, type=bool)
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_fasta_command)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

    remove_shared_memory_in_session()
