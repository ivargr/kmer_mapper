import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

from graph_kmer_index.collision_free_kmer_index import MinimalKmerIndex

import numpy as np
import argparse
from graph_kmer_index import KmerIndex
from .mapping import get_kmers_from_fasta, map_fasta
from mapper import map_kmers_to_graph_index
import time
from graph_kmer_index.index_bundle import IndexBundle
from .kmer_counting import SimpleKmerLookup
from .kmer_lookup import Advanced2
from .parser import OneLineFastaParser, Sequences
from shared_memory_wrapper import from_shared_memory, to_shared_memory, SingleSharedArray
from shared_memory_wrapper.shared_memory import remove_shared_memory_in_session
from pathos.multiprocessing import Pool
from itertools import repeat

def main():
    run_argument_parser(sys.argv[1:])

def test(args):
    print("Hi")


def map_fasta_command(args):
    if args.kmer_size > 31:
        logging.error("k must be 31 or lower")
        sys.exit(1)

    if not args.fasta_file.endswith(".fa"):
        logging.error("Only fasta files (not fq or gzipped files) are supported to the argument -f. If you have another"
                          "format, you can pipe fasta to kmer_mapper and use a dash as file name (-f -)")
        sys.exit(1)


    if args.kmer_index is None:
        if args.index_bundle is None:
            logging.error("Either a kmer index (-i) or an index bundle (-b) needs to be specified")
            sys.exit(1)
        else:
            kmer_index = IndexBundle.from_file(args.index_bundle).indexes
    else:
        if args.use_numpy:
            from .hash_table import NodeCount
            logging.info("Using numpy index")
            args.kmer_index = NodeCount.from_old_index_files(args.kmer_index)
        else:
            cls = KmerIndex
            if "minimal" in args.kmer_index:
                cls = MinimalKmerIndex
            kmer_index = cls.from_file(args.kmer_index)

    map_fasta(args, kmer_index)


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
    subparser.add_argument("-N", "--use-cython-file-reading", required=False, type=bool, default=False, help="Use cython-based file reading instead of Cython")
    subparser.add_argument("-m", "--no-shared-memory", required=False, type=bool, default=False, help="Set to True to not use shared memory for index. Increases memory usage by a factor of --n-threads.")
    subparser.add_argument("-l", "--max-read-length", required=False, type=int, default=150,
                           help="Maximum length of reads. Reads should not be longer than this.")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-r", "--include-reverse-complement", required=False, default=False, type=bool)
    subparser.add_argument("-T", "--use-two-bit-parsing", required=False, default=False, type=bool)
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_fasta_command)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

    remove_shared_memory_in_session()
