import bionumpy as bnp                                                                                                                                                                                                                 
import time


from kmer_mapper.parser import BufferedNumpyParser



t = time.perf_counter()
f = bnp.open("test.fa", buffer_type=bnp.TwoLineFastaBuffer)
for chunk in f.read_chunks(chunk_size=100000000):
    print(time.perf_counter()-t)


t = time.perf_counter()
raw_chunks = bnp.parser.NumpyFileReader(open("test.fa", "rb"), bnp.TwoLineFastaBuffer).read_chunks(chunk_size=100000000)
for chunk in raw_chunks:
    print(chunk)
    print(time.perf_counter()-t)
    print(chunk.get_sequences())
    print(time.perf_counter()-t)

"""
t = time.perf_counter()
parser = BufferedNumpyParser.from_filename("test.fa", 100000000)
for chunk in parser.get_chunks():
    print(time.perf_counter()-t)
"""