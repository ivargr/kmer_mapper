# Kmer Mapper
Python package for fast mapping of kmers from a fasta file to a [Kmer Index](https://github.com/ivargr/graph_kmer_index). Relies on Numpy and some Cython to achieve fast mapping. 

### Installation
Requires Python 3.
```bash
pip install kmer_mapper
```

### Usage
Kmer mapper works with `.fa` and `.fq` files (also gzipped). 
```bash
kmer_mapper map -i kmer_index.npz -f reads.fa -o results --n-threads 10 --kmer-size 31
```


### GPU-support (experimental)
You should have a GPU with 4 GB or more memory. You may adjust the chunk size to lower the memory usage.
```bash
kmer_mapper map -i kmer_index.npz -f reads.fa -o results --n-threads 10 --kmer-size 31 --gpu True --chunk-size 10000000
```



