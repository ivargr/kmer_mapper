from kmer_mapper.mapper import map_kmers_to_graph_index
import pytest
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


@pytest.mark.skip()
def test_map_to_kmer_index():
    node_kmers = ["ACT", "CTT", "cCG", "ATT"]
    nodes = np.arange(len(node_kmers), dtype=np.uint32)
    node_kmers = np.array([sequence_to_kmer_hash(kmer) for kmer in node_kmers], dtype=np.uint64)
    flat_kmers = FlatKmers(node_kmers, nodes, ref_offsets=np.arange(len(nodes), dtype=np.uint64))
    kmer_index = KmerIndex.from_flat_kmers(flat_kmers, modulo=21)
    kmer_index.convert_to_int32()

    assert kmer_index.get(sequence_to_kmer_hash("ccg"))[0][0] == 2

    print("Node kmers", node_kmers)
    node_counts = map_kmers_to_graph_index(kmer_index, 100, node_kmers, 1000)
    print(node_counts)


#test_get_kmers_from_fasta()
#test_get_kmers_from_fasta()
#test_map_to_index()
#test_map_to_kmer_index()



