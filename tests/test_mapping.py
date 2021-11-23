from kmer_mapper.mapping import get_kmers_from_fasta 
from kmer_mapper.mapper import map_kmers_to_graph_index
from graph_kmer_index import sequence_to_kmer_hash, letter_sequence_to_numeric
from graph_kmer_index.read_kmers import ReadKmers
from Bio import Seq
import numpy as np
import time
from graph_kmer_index import KmerIndex
from graph_kmer_index import FlatKmers

def make_dummy_fasta():
    sequences = ["AAAAATTCCACAccagAT", "AACCCAcaAAAACCCACA", "ACTACACACAACCATT", "ACACCATGGAgAGGAATTAC"]
    sequences = ["ACTATCAGCATCAGCATCAGCAGCATGCTACGACGACTACGACTACGGACGACT"]
    #sequences = ["AAAAC"] # , "AACCC", "ACT"]
    file_name = "testfasta.fa.tmp"
    f = open(file_name, "w")
    for i, seq in enumerate(sequences):
        f.write(">seq" + str(i) + "\n" + seq + "\n")

    f.close()

    return sequences, file_name


def true_kmers_from_sequences(file_name, k=3):
    kmers = ReadKmers.from_fasta_file(file_name, k=k)
    kmers = np.concatenate(list(kmers))
    return np.unique(kmers, return_counts=True)


def test_get_kmers_from_fasta():
    sequences, file_name = make_dummy_fasta()
    for k in [29]:
        true_kmers, true_counts = true_kmers_from_sequences(file_name, k=k)
        print("True: ", true_kmers)
        print("True counts: ", true_counts)

        kmers, counts = get_kmers_from_fasta("testfasta.fa.tmp", k=k, max_read_length=max([len(s) for s in sequences]))

        print("Kmers: ", kmers)
        print("Counts: ", counts)

        assert len(set(kmers)) == len(set(true_kmers))

        for kmer, count in zip(kmers, counts):
            assert kmer in true_kmers
            true_count = true_counts[true_kmers == kmer]
            assert true_count == count, "True count %d != %d" % (true_count, count)


def test_map_to_kmer_index():
    node_kmers = ["ACT", "CTT", "cCG", "ATT"]
    nodes = np.arange(len(node_kmers), dtype=np.uint32)
    node_kmers = np.array([sequence_to_kmer_hash(kmer) for kmer in node_kmers], dtype=np.uint64)
    flat_kmers = FlatKmers(node_kmers, nodes, ref_offsets=np.arange(len(nodes), dtype=np.uint64))
    kmer_index = KmerIndex.from_flat_kmers(flat_kmers, modulo=21)

    assert kmer_index.get(sequence_to_kmer_hash("ccg"))[0][0] == 2

    print("Node kmers", node_kmers)
    node_counts = map_kmers_to_graph_index(kmer_index, 100, node_kmers, 1000)
    print(node_counts)


def test_runtime():
    start = time.time()
    kmers = get_kmers_from_fasta("tests/hg002_simulated_reads_15x.fa", k=31, max_read_length=150, chunk_size=50000, return_only_kmers=True)
    end = time.time()
    print("Total time: %.4f" % (end-start))

def test_map_to_index():
    index = KmerIndex.from_file("tests/kmer_index_only_variants.npz")
    start = time.time()
    kmers = get_kmers_from_fasta("tests/hg002_simulated_reads_15x.fa", k=31, max_read_length=150, chunk_size=500000, return_only_kmers=True)
    node_counts = map_kmers_to_graph_index(index, 10000000, kmers, 1000)
    print(node_counts)
    end = time.time()
    print("Total time: %.4f" % (end-start))

test_get_kmers_from_fasta()
#test_get_kmers_from_fasta()
#test_map_to_index()
#test_map_to_kmer_index()



