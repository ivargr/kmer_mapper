from itertools import product
import numpy as np

from .parser import get_kmer_mask


class TwoBitEncoding:
    pass


class ACTGTwoBitEncoding:
    letters = ["A", "C", "T", "G"]
    bitcodes = ["00", "01", 
                "10", "11"]
    reverse = np.array([1, 3, 20, 7], dtype=np.uint8)
    _lookup_2bytes_to_4bits = np.zeros(256*256, dtype=np.uint8)

    for c1, c2 in product([0, 1, 2, 3], repeat=2):
        idx = reverse[c1]*256+reverse[c2]
        _lookup_2bytes_to_4bits[idx] = c1*4+c2
    _shift_4bits = (4*np.arange(2, dtype=np.uint8))
    _shift_2bits = 2*np.arange(4, dtype=np.uint8)

    @classmethod
    def convert_2bytes_to_4bits(cls, two_bytes):
        assert two_bytes.dtype == np.uint16, two_bytes.dtype
        return cls._lookup_2bytes_to_4bits[two_bytes]

    @classmethod
    def join_4bits_to_byte(cls, four_bits):
        return np.bitwise_or.reduce(four_bits << cls._shift_4bits, axis=1)

    @classmethod
    def complement(cls, char):
        complements = np.packbits([1, 0, 1, 0, 1, 0, 1, 0])
        dtype = char.dtype
        return (char.view(np.uint8) ^ complements).view(dtype)

    @classmethod
    def from_bytes(cls, sequence):
        assert sequence.dtype==np.uint8
        assert sequence.size % 4 == 0, sequence.size
        sequence = sequence & 31
        four_bits = cls.convert_2bytes_to_4bits(sequence.view(np.uint16))
        codes = cls.join_4bits_to_byte(four_bits.reshape(-1, 2))
        assert codes.dtype == np.uint8, codes.dtype
        return codes.flatten().view(np.uint8)

    @classmethod
    def from_string(cls, string):
        byte_repr = np.array([ord(c) for c in string], dtype=np.uint8)
        return cls.from_bytes(byte_repr)

    @classmethod
    def to_string(cls, bits):
        byte_repr = cls.to_bytes(bits)
        return "".join(chr(b) for b in byte_repr)

    @classmethod
    def to_bytes(cls, sequence):
        assert sequence.dtype==np.uint8
        bit_mask = np.uint8(3) # last two bits
        all_bytes = (sequence[:, None]>>cls._shift_2bits) & bit_mask
        return cls.reverse[all_bytes.flatten()]+96




class SimpleEncoding(ACTGTwoBitEncoding):
    _lookup_byte_to_2bits = np.zeros(256, dtype=np.uint8)
    _lookup_byte_to_2bits[[97, 65]] = 0
    _lookup_byte_to_2bits[[99, 67]] = 1
    _lookup_byte_to_2bits[[116, 84]] = 2
    _lookup_byte_to_2bits[[103, 71]] = 3

    _shift_2bits = 2*np.arange(4, dtype=np.uint8)

    @classmethod
    def convert_byte_to_2bits(cls, one_byte):
        assert one_byte.dtype == np.uint8, one_byte.dtype
        return cls._lookup_byte_to_2bits[one_byte]

    @classmethod
    def join_2bits_to_byte(cls, two_bits_vector):
        return np.bitwise_or.reduce(two_bits_vector << cls._shift_2bits, axis=-1)

    @classmethod
    def from_bytes(cls, sequence):
        assert sequence.dtype==np.uint8
        assert sequence.size % 4 == 0, sequence.size
        two_bits = cls.convert_byte_to_2bits(sequence)
        codes = cls.join_2bits_to_byte(two_bits.reshape(-1, 4))
        return codes.flatten()


class TwoBitSequences:
    encoding = ACTGTwoBitEncoding
    def __init__(self, sequences, intervals):
        self.sequences = sequences
        self.intervals = intervals

    @classmethod
    def from_byte_sequence(cls, sequences):
        twobit_sequences = cls.encoding.from_bytes(sequences.sequences)
        intervals = (sequences.intervals_start, sequences.intervals_end)
        return cls(twobit_sequences, intervals)

class TwoBitHash:
    def __init__(self, k=31, dtype=np.uint64):
        self._dtype = dtype
        n_bits_in_dtype=8*dtype(0).nbytes
        self._n_letters_in_dtype=n_bits_in_dtype//2
        self.k = k
        n_bits_to_be_moved = 2+n_bits_in_dtype-2*k
        n_masks = n_bits_in_dtype//2-k+1
        n_shifts = n_bits_in_dtype//n_bits_to_be_moved
        self._shifts = n_bits_to_be_moved*np.arange(n_shifts, dtype=dtype)
        self._masks = dtype(4**k-1) << (2*np.arange(n_masks, dtype=dtype))
        self._mask_shifts = dtype(2)*np.arange(n_masks, dtype=dtype)
        self._move_from_mask = dtype(2)**(n_bits_to_be_moved*np.arange(n_shifts, dtype=dtype))-dtype(1)
        self._moves = (dtype(n_bits_in_dtype)-dtype(n_bits_to_be_moved)*np.arange(n_shifts, dtype=dtype))
        self._move_to_mask = (self._move_from_mask << self._moves).astype(dtype)
        self._set_correct_shapes()

    def __repr__(self):
        return ">>>\n".join(str(c) for c in
                         [self._shifts, 
                          [bin(m) for m in self._masks.flatten()], 
                          self._moves,
                          [bin(m) for m in self._move_from_mask.flatten()]])

    def _set_correct_shapes(self):
        self._masks = self._masks.reshape(1, 1, -1)
        self._mask_shifts = self._mask_shifts.reshape(1, 1, -1)
        self._shifts = self._shifts.reshape(1, -1, 1)
        self._moves = self._moves.reshape(1, -1, 1)
        self._move_from_mask = self._move_from_mask.reshape(1, -1, 1)
        self._move_to_mask = self._move_to_mask.reshape(1, -1, 1)

    def np_get_kmers(self, sequence):
        assert sequence.dtype==self._dtype, sequence.dtype
        sequence = sequence.reshape(-1, 1, 1)
        shifted = (sequence >> self._shifts) # & ~self._move_to_mask
        shifted[:-1] |= (sequence[1:] & self._move_from_mask) << self._moves
        all_kmers = ((shifted & self._masks)>>self._mask_shifts)
        n_kmers = self._n_letters_in_dtype*sequence.size-self.k+1
        return all_kmers.ravel()[:n_kmers]

    def get_kmer_hashes(self, sequences):
        """Matching old interface"""
        mask = get_kmer_mask(sequences.intervals, sequences.sequences.size*4, self.k)
        kmers = self.np_get_kmers(sequences.sequences.view(self._dtype))
        forward_hashes = twobit_swap(kmers) >> ((self._n_letters_in_dtype-self.k)*2)
        reverse_hashes = sequences.encoding.complement(kmers) & self._dtype(4**self.k-1)
        return forward_hashes, reverse_hashes, mask

class SimpleTwoBitHash(TwoBitHash):
    def __init__(self, k=31, dtype=np.uint64):
        self._dtype = dtype
        self.k = k
        self._mask = dtype(4**k-1)
        self._n_letters_in_dtype = 4*dtype(0).nbytes
        self._shifts = dtype(2)*np.arange(self._n_letters_in_dtype, dtype=dtype)
        self._move_from_mask = dtype(2)**self._shifts-dtype(1)
        self._moves = self._shifts[::-1]+dtype(2)

    def np_get_kmers(self, sequence):
        assert sequence.dtype==self._dtype, sequence.dtype
        sequence = sequence.reshape(-1, 1)
        shifted = (sequence >> self._shifts)
        shifted[:-1] |= (sequence[1:] & self._move_from_mask) << self._moves
        all_kmers = (shifted & self._mask)
        n_kmers = self._n_letters_in_dtype*sequence.size-self.k+1
        return all_kmers.ravel()[:n_kmers]

class VerySimpleTwoBitHash(TwoBitHash):
    def __init__(self, k=31, dtype=np.uint64):
        self._dtype = dtype
        self.k = k
        self._mask = dtype(4**k-1)
        self._n_letters_in_dtype = 4*dtype(0).nbytes
        self._shifts = dtype(2)*np.arange(self._n_letters_in_dtype, dtype=dtype)
        self._rev_shifts = self._shifts[::-1]+dtype(2)

    def np_get_kmers(self, sequence):
        assert sequence.dtype==self._dtype, sequence.dtype
        sequence = sequence.reshape(-1, 1)
        shifted = (sequence[:-1] >> self._shifts) | (sequence[1:] << self._rev_shifts)
        all_kmers = np.append(shifted, sequence[-1]>>(self._dtype(2)*np.arange(1+self._n_letters_in_dtype-self.k, dtype=self._dtype)))  & self._mask
        n_kmers = self._n_letters_in_dtype*sequence.size-self.k+1
        assert all_kmers.size >= n_kmers, (all_kmers.size, n_kmers)
        return all_kmers.ravel()[:n_kmers]

    def np_get_kmers_with_buffer(self, sequence):
        return self._mask & (sequence[:-1, None] >> self._shifts) | (sequence[1:, None] << self._rev_shifts)


    def get_kmer_hashes(self, sequences):
        """Matching old interface"""
        last_end = sequences.intervals[1][-1]
        if sequences.sequences.size*4- last_end>=self._n_letters_in_dtype-self.k+1:
            mask = get_kmer_mask(sequences.intervals, last_end, self.k)
            kmers = self.np_get_kmers_with_buffer(sequences.sequences.view(self._dtype))[:mask.size]
            reverse_hashes = sequences.encoding.complement(kmers) & self._dtype(4**self.k-1)
            forward_hashes = twobit_swap(kmers) >> ((self._n_letters_in_dtype-self.k)*2)
            return forward_hashes, reverse_hashes, mask
        else:
            return super().get_kmer_hashes(sequences)

class FastTwoBitHash(VerySimpleTwoBitHash):
    def np_get_kmers(self, sequence):
        assert sequence.dtype==self._dtype, sequence.dtype
        result = (sequence[:, None] >> self._shifts)
        result[:-1] |= (sequence[1:, None] << self._rev_shifts)
        result &= self._mask
        n_kmers = self._n_letters_in_dtype*sequence.size-self.k+1
        return result.ravel()[:n_kmers]

    def get_new_kmer_hashes(self, sequences):
        last_end = sequences.intervals[1][-1]
        mask = get_kmer_mask(sequences.intervals, last_end, self.k)
        if sequences.sequences.size*4- last_end>=self._n_letters_in_dtype-self.k+1:
            kmers = self.np_get_kmers_with_buffer(sequences.sequences.view(self._dtype)).ravel()
        else: 
            kmers = self.np_get_kmers(sequences.sequences.view(self._dtype))
        return kmers[:mask.size][mask] & self._mask

    def np_get_kmers_with_buffer(self, sequence):
        res = (sequence[:-1, None] >> self._shifts)
        res |= (sequence[1:, None] << self._rev_shifts)
        return res

def simple_hash(s, k):
    codes = SimpleEncoding.convert_byte_to_2bits(
        np.array([ord(c) for c in s], dtype=np.uint8))
    power_array = 4**np.arange(k)[::-1]
    return np.convolve(codes, power_array, mode="valid")

def twobit_swap(number):
    dtype = number.dtype
    byte_lookup = np.zeros(256, dtype=np.uint8)
    power_array = 4**np.arange(4)
    rev_power_array = power_array[::-1]
    for two_bit_string in product([0, 1, 2, 3], repeat=4):
        byte_lookup[np.sum(power_array*two_bit_string)] = np.sum(rev_power_array*two_bit_string)
    new_bytes = byte_lookup[number.view(np.uint8)]
    return new_bytes.view(dtype).byteswap()


if __name__ == "__main__":
    k = 7
    dtype=np.uint16
    h = TwoBitHash(k=k, dtype=dtype)
    # s = "GGGGCCCCTTTTAAAA"# *4
    s = "AGCTTCGCTGCCTTGG"# *4
    #s += "GGGGCCCCGGGCCGGT"
    #s += "GGGGCCCCTTTTAAAA"# *4
    bits = ACTGTwoBitEncoding.from_string(s)
    # print([bin(b) for b in bits])
    # print(h)
    print(ACTGTwoBitEncoding.to_string(bits))
    kmers = h.np_get_kmers(bits.view(dtype))
    print([bin(k) for k in kmers])
    print(kmers)
    print("TRUTH")
    print([bin(k) for k in simple_hash(s, k=k)])
    print(simple_hash(s, k=k))
    print("----------DEBUG-------------")
    print([bin(m) for m in  h._masks[0,0]])
    print([bin(m) for m in  h._move_to_mask[0,:, 0]])
    print([bin(m) for m in  h._move_from_mask[0,:, 0]])
    print(h._mask_shifts)

    # print(dtype(4**k-1) & bits.view(dtype))
    # print(((dtype(4**k-1)<<2) & bits.view(dtype))>>2)
    #print([ACTGTwoBitEncoding.to_string(kmers[i:i+1].view(np.uint8))
    #      for i in range(kmers.size)])
