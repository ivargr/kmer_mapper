import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import sys
import argparse
from graph_kmer_index import KmerIndex
from kmer_mapper.mapping import get_kmers_from_fasta
from kmer_mapper.mapper import map_kmers_to_graph_index
import time



def main():
    run_argument_parser(sys.argv[1:])

def test(args):
    print("Hi")


def map_fasta(args):
    index = KmerIndex.from_file(args.kmer_index)
    n_nodes = index.max_node_id()
    start_time = time.time()
    logging.info("N nodes: %d" % n_nodes)
    kmers = get_kmers_from_fasta(args.fasta_file, k=args.kmer_size,
                                 max_read_length=args.max_read_length,
                                 chunk_size=args.chunk_size,
                                 return_only_kmers=True)
    node_counts = map_kmers_to_graph_index(index, n_nodes+1, kmers, 1000)
    np.save(args.output_file, node_counts)
    logging.info("Saved node counts to %s.npy" % args.output_file)
    logging.info("Spent %.3f sec in total mapping kmers" % (time.time()-start_time))


def run_argument_parser(args):

    parser = argparse.ArgumentParser(
        description='Kmer Mapper',
        prog='kmer_mapper',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("map")
    subparser.add_argument("-i", "--kmer-index")
    subparser.add_argument("-f", "--fasta-file")
    subparser.add_argument("-k", "--kmer-size", required=False, default=31, type=int)
    subparser.add_argument("-t", "--n-threads", required=False, default=16, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, type=int, default=500000, help="N reads to process in each chunk")
    subparser.add_argument("-l", "--max-read-length", required=False, type=int, default=150,
                           help="Maximum length of reads. Reads should not be longer than this.")
    subparser.add_argument("-o", "--output-file")


    subparser.set_defaults(func=map_fasta)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
