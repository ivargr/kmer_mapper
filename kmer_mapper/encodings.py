from itertools import product
import numpy as np

class TwoBitEncoding:
    pass

class TwoBitSequence:
    def __init__():
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

class TwoBitHash:
    first_2 = np.uint64(2**63+2**62)
    last_31 = ~first_2
    last_2 = np.uint64(3)
    first_31 = ~last_2
    
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
        print("##########>>>", [bin(s) for s in sequence])
        sequence = sequence.reshape(-1, 1, 1)
        assert sequence.dtype==self._dtype, sequence.dtype
        shifted = (sequence >> self._shifts) & ~self._move_to_mask
        added = (sequence[1:] & self._move_from_mask) << self._moves
        shifted[:-1] |= added
        return ((shifted & self._masks)>>self._mask_shifts).flatten()[:self._n_letters_in_dtype*sequence.size-self.k+1]

    def get_kmers(self, sequence):
        assert sequence.dtype==np.uint8, sequence.dtype
        buffers = [sequences.copy(), np.empty_like(sequences)]
        kmers = np.empty(sequence.size*31, dtype=np.uint64).reshape(32, -1)
        kmers[0] = sequence
        last_buffer = kmers[-1]
        for i in range(32):
            kmers[i+1] = kmers[i]<<2
            kmers[i+1][:-1] |= kmers[i][1:] & self.first_2
        hashes = kmers & last_31

def simple_hash(s, k):
    codes = SimpleEncoding.convert_byte_to_2bits(
        np.array([ord(c) for c in s], dtype=np.uint8))
    power_array = 4**np.arange(k)[::-1]
    print("->", np.sum(power_array*codes[:power_array.size]))
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
