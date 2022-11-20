import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


import os
import time
import tqdm
from .util import _get_kmer_index_from_args, get_kmer_hashes_from_chunk_sequence, open_file
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import pyximport
pyximport.install()

from graph_kmer_index.collision_free_kmer_index import CounterKmerIndex
import argparse
from graph_kmer_index import KmerIndex
from shared_memory_wrapper.shared_memory import remove_shared_memory_in_session, remove_shared_memory
from shared_memory_wrapper.shared_memory import get_shared_pool
from shared_memory_wrapper import object_to_shared_memory, object_from_shared_memory
import bionumpy as bnp
from kmer_mapper.mapper import map_kmers_to_graph_index
from shared_memory_wrapper.shared_array_map_reduce import additative_shared_array_map_reduce


def main():
    run_argument_parser(sys.argv[1:])


def map_cpu(args, kmer_index, chunk_sequence_name):
    """Maps a sequence stored in shared memory to the kmer index"""
    kmer_size = args["kmer_size"]
    logging.debug("Starting _mapper with chunk %s" % chunk_sequence_name)
    t = time.perf_counter()
    chunk_sequence = object_from_shared_memory(chunk_sequence_name).get_data().sequence
    logging.debug("N sequences in chunk: %d" % len(chunk_sequence))
    hashes = get_kmer_hashes_from_chunk_sequence(chunk_sequence, kmer_size)

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


def map_gpu(index, chunks, k, map_reverse_complements=False):
    """Maps a generator of chunks to the index (no shared memory)"""
    from .gpu_counter import GpuCounter
    logging.info("Making counter")
    counter = GpuCounter.from_kmers_and_nodes(index._kmers, index._nodes, k)
    counter.initialize_cuda(130000001)
    logging.info("CUDA counter initialized")

    t_start = time.perf_counter()
    for i, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        hashes = get_kmer_hashes_from_chunk_sequence(chunk.get_data().sequence, k)
        logging.info("Time to get hashes for chunk ", (time.perf_counter()-t0))
        t1 = time.perf_counter()
        counter.count(hashes, count_revcomps=map_reverse_complements)
        logging.info("Time to count %d hashes for chunk: %.10f" % (len(hashes), time.perf_counter()-t1))
        logging.info("GPU: Whole chunk finished in ", (time.perf_counter()-t0))

    logging.info("Time spent only on hashing and counting hashes: %.5f" % (time.perf_counter()-t_start))
    return counter.get_node_counts(min_nodes=index.max_node_id())


def map_bnp(args):
    if args.debug:
        logging.info("Will print debug log")
        logging.getLogger().setLevel(logging.DEBUG)

    k = args.kmer_size
    kmer_index = _get_kmer_index_from_args(args)

    start_time = time.perf_counter()
    n_bytes = os.stat(args.reads).st_size
    approx_number_of_chunks = int(n_bytes / args.chunk_size)

    if args.gpu:
        import cupy as cp
        bnp.set_backend(cp)
        file = open_file(args.reads)
        chunks = file.read_chunks(min_chunk_size=args.chunk_size)
        node_counts = map_gpu(kmer_index, chunks, k, args.map_reverse_complements)
    else:
        assert not args.map_reverse_complements, "Mapping reverse complements only supported with GPU-mode for now"
        file = open_file(args.reads)
        chunks = (object_to_shared_memory(raw_chunk) for
                  raw_chunk in file.read_chunks(min_chunk_size=args.chunk_size))
        chunks = tqdm.tqdm(chunks, total=approx_number_of_chunks)

        if isinstance(kmer_index, KmerIndex):
            initial_data = np.zeros(kmer_index.max_node_id()+1)
        else:
            initial_data = np.zeros_like(kmer_index.counter._values)

        args_dict = vars(args)
        args_dict.pop("func")
        t_before_map = time.perf_counter()
        node_counts = additative_shared_array_map_reduce(map_cpu,
                                                         chunks,
                                                         initial_data,
                                                         (args_dict, kmer_index),
                                                         n_threads=args.n_threads
                                                         )
        logging.info("Time spent only on hashing and counting hashes: %.4f" % (time.perf_counter()-t_before_map))

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
    subparser.add_argument("-g", "--gpu", default=False, type=bool,
                           help="Set to True to use GPU-counting. Experimental."
                           " Requires suitable hardware and dependencies.")
    subparser.add_argument("-r", "--map-reverse-complements", default=False, type=bool,
                            help="Also count kmers in reverse complement of reads. "
                                 "Default False. Not necessary if index contains reverse complements.")
    subparser.set_defaults(func=map_bnp)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

    remove_shared_memory_in_session()
