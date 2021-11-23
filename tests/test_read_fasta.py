import numpy as np
np.set_printoptions(suppress=True)
import pyximport; pyximport.install()
from kmer_mapper import mapper
from kmer_mapper import util
import time

mapper.test()

chunks = mapper.read_fasta_into_chunks("tests/hg002_simulated_reads_15x.fa", 100)

for chunk in chunks:
    print(chunk)

functions = [ mapper.get_count]

for function in functions:
    start = time.time()
    res = function("tests/hg002_simulated_reads_15x.fa")
    print(res)
    print("Took %.4f sec" % (time.time()-start))
