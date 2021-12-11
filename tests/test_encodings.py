from kmer_mapper.encodings import ACTGTwoBitEncoding
from kmer_mapper.encodings import SimpleEncoding, TwoBitHash, twobit_swap, TwoBitSequences, SimpleTwoBitHash, VerySimpleTwoBitHash, FastTwoBitHash
from kmer_mapper.parser import Sequences, KmerHash
import numpy as np
Encoding = ACTGTwoBitEncoding
# SimpleEncoding
TwoBitHash=FastTwoBitHash
def test_simple():
    sequence = np.array([97, 99, 116, 103], dtype="uint8")
    bits = np.unpackbits(Encoding.from_bytes(sequence))
    assert bits.tolist()==[1, 1,
                           1, 0,
                           0, 1,
                           0, 0]

def test_forward_and_back():
    sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67],
                        dtype="uint8")
    new_sequence = Encoding.to_bytes(Encoding.from_bytes(sequence))
    assert np.all((sequence & 31)==new_sequence & 31)

def test_long_forward_and_back():
    sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67,
                         116, 99, 116, 67, 67, 65, 67, 103,
                         116, 103, 65, 65, 65, 65, 67, 103],
                        dtype="uint8")
    new_sequence = Encoding.to_bytes(Encoding.from_bytes(sequence))
    print(new_sequence & 31)
    print(sequence & 31)
    assert np.all((sequence & 31)==new_sequence & 31)

def test_view_corresponds_to_hash():
    sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67],
                        dtype="uint8")
    two_bits = SimpleEncoding.convert_byte_to_2bits(sequence)
    power_array = np.power(4, np.arange(8, dtype=np.uint16), dtype=np.uint16)
    h = np.sum(two_bits*power_array)
    bit_sequence = Encoding.from_bytes(sequence)
    assert bit_sequence.view(np.uint16) == h, (bin(bit_sequence.view(np.uint16)[0]), bin(h))

def test_view_corresponds_to_32hash():
    sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67],
                        dtype="uint8")
    two_bits = SimpleEncoding.convert_byte_to_2bits(sequence)
    power_array = np.power(4, np.arange(16, dtype=np.uint32), dtype=np.uint32)
    h = np.sum(two_bits*power_array)
    bit_sequence = Encoding.from_bytes(sequence)
    assert bit_sequence.view(np.uint32) == h, (bin(bit_sequence.view(np.uint32)[0]), bin(h))

def test_hashing():
    k = 31
    dtype=np.uint64
    np.random.seed(100)
    sequence = np.random.choice([97, 99, 116, 103, 65, 67, 71, 84], 64).astype(np.uint8)
    _sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67,
                         116, 99, 116, 67, 67, 65, 67, 103,
                         116, 103, 65, 65, 65, 65, 67, 103],
                        dtype="uint8")
    codes = SimpleEncoding.convert_byte_to_2bits(sequence)
    power_array = 4**np.arange(k, dtype=np.uint64)[::-1]
    true_kmers = np.convolve(codes, power_array, mode="valid")
    bits = Encoding.from_bytes(sequence)
    h = TwoBitHash(k=k, dtype=dtype)
    kmers = h.np_get_kmers(bits.view(dtype))
    print(kmers)
    print(true_kmers)
    assert np.all(kmers==true_kmers)

def test_twobit_swap():
    sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67,
                         116, 99, 116, 67, 67, 65, 67, 103,
                         116, 103, 65, 65, 65, 65, 67, 103],
                        dtype="uint8")

    _sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67],
                        dtype="uint8")
    twobits = Encoding.from_bytes(sequence)
    print(twobits)
    reverse_twobits = Encoding.from_bytes(sequence[::-1])
    print(reverse_twobits)
    assert twobit_swap(twobits.view(np.uint64)) == reverse_twobits.view(np.uint64)

def test_hash_with_complement():
    k = 31
    dtype=np.uint64
    np.random.seed(100)
    sequence = np.random.choice([97, 99, 116, 103, 65, 67, 71, 84], 32).astype(np.uint8)
    _sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67,
                         116, 99, 116, 67, 67, 65, 67, 103,
                         116, 103, 65, 65, 65, 65, 67, 103],
                        dtype="uint8")
    codes = SimpleEncoding.convert_byte_to_2bits(sequence)
    s = Sequences(sequence, np.array([0]), np.array([codes.size]))
    true_kmers, true_rev_kmers, true_mask = KmerHash(k).get_kmer_hashes(s)
    two_bit_sequences = TwoBitSequences.from_byte_sequence(s)
    # bits = Encoding.from_bytes(sequence)
    h = TwoBitHash(k=k, dtype=dtype)
    kmers, rev_kmers, mask = h.get_kmer_hashes(two_bit_sequences)
    # kmers = h.np_get_kmers(bits.view(dtype))
    # rev_kmers = Encoding.complement(kmers) & dtype(4**k-1)
    # kmers = twobit_swap(kmers)>>2
    assert np.all(np.sort(kmers)==np.sort(true_kmers))
    assert np.all(np.sort(rev_kmers)==np.sort(true_rev_kmers))
    assert np.all(mask==true_mask)
                               
if __name__ == "__main__":
    test_simple()



