from kmer_mapper.mapper import read_fasta_into_chunks
import numpy as np
import time
from kage.reads import read_chunks_from_fasta


t = time.time()
chunk, mask = next(read_fasta_into_chunks("tests/hg002_simulated_reads_15x.fa", 500000, process_reads=False))
#chunk = next(read_chunks_from_fasta("tests/hg002_simulated_reads_15x.fa", 500000, write_to_shared_memory=True))

print("Time to read chunk and mask: %.3f" % (time.time()-t))
print(chunk)

t = time.time()
chunk = np.frombuffer(chunk, dtype=np.uint8).reshape((len(chunk), 150))
print("Time to read from buffer: %.4f" % (time.time()-t))

print(chunk)