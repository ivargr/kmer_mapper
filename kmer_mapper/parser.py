<<<<<<< HEAD
from shared_memory_wrapper import SingleSharedArray, to_shared_memory, from_shared_memory
=======
>>>>>>> f62737bb653286c253b35bb1accce23200f40f9a
import numpy as np
HEADER = 62
NEWLINE = 10
letters = ["A", "C", "T", "G"]

to_text = lambda x: "".join(chr(r) for r in x)

def get_mask_from_intervals(intervals, size):
    mask_changes = np.zeros(size+1, dtype="bool")
    mask_changes[intervals[0]]=True
    mask_changes[intervals[1]]^=True
    mask = np.logical_xor.accumulate(mask_changes)
    return mask[:-1]

<<<<<<< HEAD

class Sequences:
    def __init__(self, sequences, intervals_start, intervals_end):
        self.sequences = sequences
        self.offsets = intervals_start[1:]
        self.intervals_start = intervals_start
        self.intervals_end = intervals_end

    def __len__(self):
        return len(self.offsets)+1

    def __getitem__(self, i):
        start, end = (0, len(self.sequences))
        if i > 0:
            start = self.offsets[i-1]
        if i < len(self.offsets):
            end = self.offsets[i]
        return self.sequences[start:end]

    def __repr__(self):
        return "Seqs(%s, %s)" % (to_text(self.sequences), self.offsets)


class TextParser:
    def __init__(self, filename, chunk_size=100):
        self._file_obj = open(filename, "rb")
        self._chunk_size = chunk_size
        self._is_finished = False
        self._last_header_idx = -1

    def read_raw_chunk(self):
        array = np.empty(self._chunk_size, dtype="uint8")
        bytes_read = self._file_obj.readinto(array)
        return array, bytes_read

    def parse(self, as_shared_memory_object=False):
        while not self._is_finished:
            chunk = self.parse_chunk()
            if chunk is not None:
                if as_shared_memory_object:
                    shared_memory_name = str(np.random.randint(0, 10e15))
                    to_shared_memory(chunk, shared_memory_name)
                    yield shared_memory_name
                else:
                    yield chunk


    def parse_chunk(self):
        a, bytes_read = self.read_raw_chunk()
        self._is_finished = bytes_read < self._chunk_size
        if bytes_read == 0:
            return None
        sequences = self.get_sequences(a[:bytes_read])
        if not self._is_finished:
            self._file_obj.seek(self._last_header_idx-self._chunk_size, 1)
        return sequences

class OneLineFastaParser(TextParser):
    HEADER = 62
    n_lines_per_entry = 2
    def _cut_array(self, array, last_newline):
        self._last_header_idx = last_newline+1
        return array[:last_newline]

    def get_sequences(self, array):
        # print("<--", to_text(array), "-->")
        # print(array.size, array[-1], self._is_finished)
        assert array[0]==self.HEADER
        new_lines = np.flatnonzero(array==NEWLINE)
        if self._is_finished and array[-1] != NEWLINE:
            #Add a trailing newline if this is the last chunk
            new_lines = np.append(new_lines, array.size) 

        # Find which lines are sequences, and cut the array so that it ends with a complete sequence
        is_sequence_line = array[new_lines[:-1]+1] != self.HEADER
        idx_last_sequence_line = np.flatnonzero(is_sequence_line)[-1]
        new_lines = new_lines[:idx_last_sequence_line+2]
        is_sequence_line = is_sequence_line[:idx_last_sequence_line+1]
        array = self._cut_array(array, new_lines[-1])

        #Create a mask for where the sequences ara and move sequences to continous array
        sequence_starts = new_lines[:-1][is_sequence_line]+1
        sequence_ends = new_lines[1:][is_sequence_line]
        # intervals = new_lines.reshape(-1, 2)[is_sequence_line]
        # sequence_intervals = (intervals[:, 0]+1, intervals[:, 1])
        mask = get_mask_from_intervals((sequence_starts, sequence_ends), array.size)
        removed_areas = np.cumsum(sequence_starts-np.insert(sequence_ends[:-1], 0, 0))
        new_intervals = (sequence_starts-removed_areas, sequence_ends-removed_areas)
        # print(new_intervals)
        # print(array[mask])
        return Sequences(array[mask], new_intervals[0], new_intervals[1])


class FastaParser(TextParser):
    def _get_sequence_offsets(self, new_lines, is_header):
        header_starts = new_lines[:-1][is_header[:-1]]
        header_ends = new_lines[1:][is_header[:-1]]
        header_lens = header_ends-header_starts
        masked_count = np.ones(new_lines.size, dtype="int")
        masked_count[1:][is_header[:-1]] = header_lens
        masked_count[0] = new_lines[0]+1
        cum_masked_count = np.cumsum(masked_count)
        indexes_after_masking = new_lines+1-cum_masked_count
        starts = indexes_after_masking[1:][is_header[:-1]]
        if not self._is_finished:
            starts = starts[:-1]
        return starts

    def get_sequences(self, array):
        assert array[0] == HEADER
        all_new_lines = np.flatnonzero(array==NEWLINE)
        new_lines = all_new_lines if all_new_lines[-1] != array.size-1 else all_new_lines[:-1]
        is_header = array[new_lines+1] == HEADER
        mask_changes = np.zeros(array.size, dtype="bool")
        mask_changes[new_lines+1] = np.where(is_header, 1, 0)
        mask_changes[new_lines[1:]] = np.where(is_header[:-1], 1, 0)
        mask_changes[0] = 1
        mask_changes[new_lines[0]] = 1
        mask = np.logical_xor.accumulate(mask_changes)
        mask[all_new_lines] = 1
        beginnings = self._get_sequence_offsets(new_lines, is_header)
        intervals = (np.insert(beginnings, 0, 0), np.append(beginnings, np.count_nonzero(~mask)))
        if self._is_finished:
            return Sequences(array[~mask], intervals)
        assert np.any(is_header), ("Fasta entry to big for chunk size", to_text(array), chunk_size)
        self._last_header_idx = new_lines[is_header][-1]+1
        intervals = (np.insert(beginnings, 0, 0), np.append(beginnings, np.count_nonzero(~mask[:self._last_header_idx])))
        return Sequences(array[:self._last_header_idx][~mask[:self._last_header_idx]], intervals)


class KmerHash:
    CODES = np.zeros(256, dtype=np.uint64)
    CODES[[97, 65]] = 0
    CODES[[99, 67]] = 1
    CODES[[116, 84]] = 2
    CODES[[103, 71]] = 3

    def __init__(self, k=31):
        self.k=k
        self.POWER_ARRAY = np.power(4, np.arange(k, dtype=np.uint64), dtype=np.uint64)
        self.REV_POWER_ARRAY = np.power(4, np.arange(k, dtype=np.uint64)[::-1], dtype=np.uint64)


    def _to_text(self, seq):
        return "".join(letters[n] for n in seq)

    def get_mask(self, offsets, size):
        """
        k=3
        offsets=[5, 9]
        0123456789
        ACGTTGATTCGGA
        -----s---s---
        ---          0
         ---         1
          ---        2
           --x       3 = 5-k+1
            -x-      4 
             ---     5 = 5 
              ---    6
        """

        mask_diffs = np.zeros(size+1, dtype="int")
        mask_diffs[np.maximum(offsets-self.k+1, 0)] = 1
        mask_diffs[np.minimum(offsets, size)] = -1
        return np.cumsum(mask_diffs)[:-1]==0

    def _get_kmers(self, seq):
        return [self._to_text(seq[i:i+self.k]) for i in range(len(seq)-self.k+1)]

    def _get_codes2(self, seq):
        idxs = np.digitize((seq-65) % 32 , np.array([65, 71, 97, 103]), right=True)
        return np.asarray(idxs^(idxs>>1), dtype=np.uint64)

    def _get_codes(self, seq):
        return self.CODES[seq]

    def _join(self, kmers1, kmers2, mask):
        return np.concatenate((kmers1[mask], kmers2[mask]))

    def get_kmer_hashes(self, sequences):
        codes = self._get_codes(sequences.sequences)
        assert codes.dtype==np.uint64, codes.dtype
        kmers = np.convolve(codes, self.POWER_ARRAY, mode="valid")
        reverse_kmers = np.convolve((codes+2) % 4, self.REV_POWER_ARRAY, mode="valid")
        mask = self.get_mask(sequences.offsets, kmers.size)
        return kmers, reverse_kmers, mask # [mask], reverse_kmers[mask]

def simulate_fasta(n_seqs, seq_len, filename):
    f = open(filename, "w")
    for i in range(n_seqs):
        seq = "".join(letters[n] for n in np.random.randint(0, 4, seq_len))
        f.write(f">h{i}\n{seq}\n")
    f.close()

def main(filename, chunk_size=500*10**5, do_print=False):
    parser = OneLineFastaParser(filename, chunk_size)
    for i, seqs in enumerate(parser.parse()):
        if do_print:
            for i in range(len(seqs)):
                print(to_text(seqs[i]))
        # print(f"Parsing chunk {i}")
        kmers = KmerHash(3).get_kmer_hashes(seqs)

if __name__ == "__main__":
    import sys
    import cProfile
    #simulate_fasta(10, 20, "small.fa")
    #cProfile.run('main("large.fa")')
    # exit()
    main(sys.argv[1], int(sys.argv[2]), do_print=True)
    exit()
    # for seqs in :# parse_fasta(, 100):
    #     print(seqs)
    #     kmers = KmerHash(3).get_kmer_hashes(seqs)
    #     continue
    #     t, starts = (seqs.sequencegs, seqs.offsets)
    #     print(to_text(t), starts)
    #     print("<", to_text(t[:starts[0]]))
    #     for s,e in zip(starts[:-1], starts[1:]):
    #         print("-", to_text(t[s:e]))
    #     print(">", to_text(t[starts[-1]:]))
