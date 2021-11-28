import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
import numpy as np
import argparse
from graph_kmer_index import KmerIndex
from kmer_mapper.mapping import get_kmers_from_fasta, map_fasta
from kmer_mapper.mapper import map_kmers_to_graph_index
import time



def main():
    run_argument_parser(sys.argv[1:])

def test(args):
    print("Hi")


def map_fasta_command(args):
    map_fasta(args.kmer_index, args.fasta_file, args.chunk_size, args.n_threads, args.max_read_length, args.k)


def run_argument_parser(args):

    parser = argparse.ArgumentParser(
        description='Kmer Mapper',
        prog='kmer_mapper',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("map")
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-f", "--fasta-file", required=True)
    subparser.add_argument("-k", "--kmer-size", required=False, default=31, type=int)
    subparser.add_argument("-t", "--n-threads", required=False, default=16, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, type=int, default=500000, help="N reads to process in each chunk")
    subparser.add_argument("-l", "--max-read-length", required=False, type=int, default=150,
                           help="Maximum length of reads. Reads should not be longer than this.")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-r", "--ignore-reverse-complement", required=False, default=False, type=bool)
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_fasta)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
