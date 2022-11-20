import bionumpy as bnp                                                                                                                                                                                                                 
import time


t = time.perf_counter()
f = bnp.open("test.fa", buffer_type=bnp.TwoLineFastaBuffer)
for chunk in f.read_chunks(min_chunk_size=100000000):
    print(time.perf_counter()-t)


t = time.perf_counter()
raw_chunks = bnp.io.parser.NumpyFileReader(open("test.fa", "rb"), bnp.TwoLineFastaBuffer).read_chunks(min_chunk_size=100000000)
for chunk in raw_chunks:
    print(chunk)
    print(time.perf_counter()-t)
    print(chunk.get_data().sequence)
    print(time.perf_counter()-t)
