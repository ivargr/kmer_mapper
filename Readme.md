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
kmer_mapper map -i kmer_index.npz -f reads.fa -o results --n-threads 10 -k 31
```

Note: The newest version of Kmer mapper uses `bionumpy` for parsing and reading files. Reading gzipped files is still a bit experimental. If performance problems occur, we suggest using with non-gzipped files.