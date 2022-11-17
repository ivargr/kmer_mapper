import bionumpy as bnp
from graph_kmer_index import KmerIndex
from graph_kmer_index.kmer_hashing import kmer_hashes_to_complement_hashes, kmer_hashes_to_reverse_complement_hash
import numpy as np
from bionumpy.encodings._legacy_encodings import ACTGTwoBitEncoding


def old_hashing(sequences, kmer_size=31):
    #hashes = bnp.kmers.fast_hash(bnp.as_encoded_array(sequences, bnp.encodings.alphabet_encoding.ACTGEncoding), kmer_size).ravel()
    hashes = bnp.sequence.get_kmers(bnp.as_encoded_array(sequences, bnp.encodings.alphabet_encoding.ACTGEncoding), kmer_size).ravel().raw()
    hashes = ACTGTwoBitEncoding.complement(hashes) & np.uint64(4 ** kmer_size - 1)
    return hashes


def new_hashing(sequences, kmer_size=31):
    #hashes = bnp.kmers.fast_hash(bnp.as_encoded_array(sequences, bnp.encodings.alphabet_encoding.ACTGEncoding), kmer_size).ravel()
    hashes = bnp.sequence.get_kmers(bnp.as_encoded_array(sequences, bnp.encodings.alphabet_encoding.ACTGEncoding), kmer_size).ravel().raw()
    return hashes


def get_sequences(file="test.fa"):
    sequence = bnp.open(file).read_chunk(min_chunk_size=5000000).sequence
    return sequence



index = KmerIndex.from_file("kmer_index_only_variants_with_revcomp.npz")
kmers = index._kmers
"""
reverse_kmers = kmer_hashes_to_complement_hashes(kmers, k=31)


print(kmers)
print(reverse_kmers)

original = kmer_hashes_to_complement_hashes(reverse_kmers, k=31)
print(original)

assert np.all(original == kmers)
"""

sequence = get_sequences()
old = old_hashing(sequence)
new = new_hashing(sequence)

print(old)
print(new)
print(kmer_hashes_to_complement_hashes(new, 31))


#reverse_complement_kmers = kmer_hashes_to_reverse_complement_hash(kmers, k=31)
#n = np.unique(np.concatenate([kmers, reverse_complement_kmers]))
#print(len(n), len(kmers))


