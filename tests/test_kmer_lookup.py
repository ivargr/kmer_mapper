from kmer_mapper.kmer_lookup import *
import pytest
import numpy as np

@pytest.fixture
def kmer_lookup():
    kmers = np.array(   [7,    2, 3, 11, 5])
    node_ids = np.array([0, 3, 1, 1, 4, 5])
    representative_kmers = [7, 
                            2, 3, 
                            7,
                            11,
                            5]
    lookup = np.array([[0, 1],
                       [1, 3],
                       [3, 3],
                       [3, 4],
                       [4, 5],
                       [5, 6]])

    _lookup = np.array([[0, 2],
                       [2, 3],
                       [3, 4],
                       [4, 5],
                       [5, 6]])

    kmer_lookup = KmerLookup(kmers, representative_kmers, lookup)
    kmer_lookup.index_kmers()
    return kmer_lookup

@pytest.fixture
def simple_kmer_lookup():
    kmers = np.array(   [7,    2, 3, 11, 5])
    node_ids = np.array([0, 3, 1, 1, 4, 5])
    representative_kmers = [7, 
                            2, 3, 
                            7,
                            11,
                            5]
    lookup = np.array([0, 1, 1, 3, 4, 5])

    kmer_lookup = Advanced2(kmers, representative_kmers, lookup)
    kmer_lookup.index_kmers()
    return kmer_lookup


@pytest.fixture
def sample_kmers():
    return np.array([7, 5, 11, 3, 11, 9])

def test_count_kmers(kmer_lookup, sample_kmers):
    counts = kmer_lookup.count_kmers(sample_kmers)
    assert np.all(counts == [0, 1, 1, 1, 2])

def test_get_node_counts(kmer_lookup, sample_kmers):
    counts = kmer_lookup.get_node_counts(sample_kmers)
    assert np.all(counts == [1, 1, 0, 1, 2, 1])

def test_simple_get_node_counts(simple_kmer_lookup, sample_kmers):
    counts = simple_kmer_lookup.get_node_counts(sample_kmers)
    print(counts)
    assert np.all(counts == [1, 1, 0, 1, 2, 1])
