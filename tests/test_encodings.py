from kmer_mapper.encodings import ACTGTwoBitEncoding
from kmer_mapper.encodings import SimpleEncoding, TwoBitHash
import numpy as np
Encoding = ACTGTwoBitEncoding
# SimpleEncoding
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
    sequence = np.random.choice([97, 99, 116, 103, 65, 67, 71, 84], 256).astype(np.uint8)
    _sequence = np.array([97, 99, 116, 103, 116, 65, 67, 67,
                         99, 99, 99, 65, 116, 116, 67, 67,
                         116, 99, 116, 67, 67, 65, 67, 103,
                         116, 103, 65, 65, 65, 65, 67, 103],
                        dtype="uint8")
    codes = SimpleEncoding.convert_byte_to_2bits(sequence)
    power_array = 4**np.arange(k)[::-1]
    true_kmers = np.convolve(codes, power_array, mode="valid")
    bits = Encoding.from_bytes(sequence)
    h = TwoBitHash(k=k, dtype=dtype)
    kmers = h.np_get_kmers(bits.view(dtype))
    assert np.all(kmers==true_kmers)
                               
if __name__ == "__main__":
    test_simple()



