import gzip
import logging
import os
import sys
import time
from pathlib import PurePath
import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import pyximport
pyximport.install()

from graph_kmer_index.collision_free_kmer_index import MinimalKmerIndex, CounterKmerIndex
import argparse
from graph_kmer_index import KmerIndex
from .mapping import map_fasta
from graph_kmer_index.index_bundle import IndexBundle
from shared_memory_wrapper.shared_memory import remove_shared_memory_in_session, remove_shared_memory
from shared_memory_wrapper.shared_memory import get_shared_pool
from shared_memory_wrapper import from_file, object_to_shared_memory, object_from_shared_memory
import bionumpy as bnp
from kmer_mapper.mapper import map_kmers_to_graph_index
from bionumpy.encodings import ACTGTwoBitEncoding, ACTGEncoding
from shared_memory_wrapper.shared_array_map_reduce import additative_shared_array_map_reduce
import npstructures


# modified version of bionumpy's fast_hash
# will use bionumpy's fast hash in the future
@bnp.util.convolution
def fast_hash(sequence, k, encoding=None):
    sequence = bnp.sequences.as_encoded_sequence_array(sequence, ACTGEncoding)
    if encoding:
        sequence = encoding.encode(sequence)

    bit_array = npstructures.bitarray.BitArray.pack(sequence, bit_stride=2)
    hashes = bit_array.sliding_window(k)
    return hashes




def main():
    run_argument_parser(sys.argv[1:])


def _get_kmer_index_from_args(args):
    # allowing multiple ways to specify kmer index, try to get the index
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
        try:
            kmer_index = cls.from_file(args.kmer_index)
            kmer_index.convert_to_int32()
            kmer_index.remove_ref_offsets()  # not needed, will save us some memory
        except:
            kmer_index = from_file(args.kmer_index)
            assert isinstance(kmer_index, CounterKmerIndex)
            logging.info("Kmer index is counter index")

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


def _mapper(kmer_size, kmer_index, chunk_sequence_name):
    logging.debug("Starting _mapper with chunk %s" % chunk_sequence_name)
    t = time.perf_counter()
    chunk_sequence = object_from_shared_memory(chunk_sequence_name).get_sequences()
    logging.debug("N sequences in chunk: %d" % len(chunk_sequence))
    hashes = fast_hash(chunk_sequence, kmer_size, encoding=None).ravel()
    hashes = ACTGTwoBitEncoding.complement(hashes) & np.uint64(4 ** kmer_size - 1)
    logging.debug("N hashes: %d" % len(hashes))

    t0 = time.perf_counter()
    if isinstance(kmer_index, CounterKmerIndex):
        kmer_index.counter.count(hashes)
        mapped = kmer_index.counter._values
        logging.debug("Mapped with counter. Got values of length %d" % len(mapped))
    else:
        mapped = map_kmers_to_graph_index(kmer_index, kmer_index.max_node_id(), hashes)

    logging.debug("Mapping %d hashes took %.3f sec" % (len(hashes), time.perf_counter()-t0))

    remove_shared_memory(chunk_sequence_name)
    logging.debug("Chunk of %d reads took %.2f sec" % (len(chunk_sequence), time.perf_counter()-t))
    return mapped


def open_file(filename):
    path = PurePath(filename)
    suffix = path.suffixes[-1]
    try:
        buffer_type = bnp.files._get_buffer_type(suffix)
    except RuntimeError:
        logging.error("Unsupported file suffix %s" % suffix)
        raise
    
    if path.suffixes[-1] == ".fa":
        # override buffer type for some performance gain
        buffer_type = bnp.TwoLineFastaBuffer
        logging.info("Using buffer type TwoLineFastaBuffer")

    open_func = gzip.open if suffix == ".gz" else open
    return bnp.parser.NumpyFileReader(open_func(filename, "rb"), buffer_type)


def map_bnp(args):
    if args.debug:
        logging.info("Will print debug log")
        logging.getLogger().setLevel(logging.DEBUG)

    k = args.kmer_size
    kmer_index = _get_kmer_index_from_args(args)

    start_time = time.perf_counter()

    file = open_file(args.reads)

    n_bytes = os.stat(args.reads).st_size
    approx_number_of_chunks = int(n_bytes / args.chunk_size)

    chunks = tqdm.tqdm((object_to_shared_memory(raw_chunk) for
        raw_chunk in file.read_chunks(chunk_size=args.chunk_size)), total=approx_number_of_chunks)

    if isinstance(kmer_index, KmerIndex):
        initial_data = np.zeros(kmer_index.max_node_id()+1)
    else:
        initial_data = np.zeros_like(kmer_index.counter._values)

    node_counts = additative_shared_array_map_reduce(_mapper,
                                                     chunks,
                                                     initial_data,
                                                     (args.kmer_size, kmer_index),
                                                     n_threads=args.n_threads
                                                     )

    if isinstance(kmer_index, CounterKmerIndex):
        # node counts is not node counts, but kmer counts, get node counts
        t = time.perf_counter()
        kmer_index.counter._values = node_counts
        node_counts = kmer_index.get_node_counts()
        logging.info("Time spent getting node counts in the end: %.3f" % (time.perf_counter()-t))


    np.save(args.output_file, node_counts)
    logging.info("Saved node counts to %s.npy" % args.output_file)
    logging.info("Spent %.3f sec in total mapping kmers using %d threads" % (time.perf_counter()-start_time, args.n_threads))
    remove_shared_memory_in_session()



def run_argument_parser(args):

    parser = argparse.ArgumentParser(
        description='Kmer Mapper',
        prog='kmer_mapper',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("legacy_map", help="Legacy mapper kept for backwards compatibility")
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



    subparser = subparsers.add_parser("map", help="Map reads to a kmer index")
    subparser.add_argument("-i", "--kmer-index", required=False)
    subparser.add_argument("-b", "--index-bundle", required=False)
    subparser.add_argument("-f", "--reads", required=True, help="Reads in .fa, .fq, .fa.gz, or fq.gz format")
    subparser.add_argument("-k", "--kmer-size", required=False, default=31, type=int)
    subparser.add_argument("-t", "--n-threads", required=False, default=16, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, type=int, default=100000000,
                           help="N bytes to process in each chunk")
    subparser.add_argument("-o", "--output-file", required=True)
    subparser.add_argument("-d", "--debug", required=False, help="Set to True to print debug log")
    subparser.add_argument("-I", "--max-hits-per-kmer", required=False, default=1000, type=int,
                           help="Ignore kmers that have more than this amount of hits in index")
    subparser.set_defaults(func=map_bnp)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

    remove_shared_memory_in_session()
