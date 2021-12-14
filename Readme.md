# Kmer Mapper
Python package for fast mapping of kmers from a fasta file to a [Kmer Index](https://github.com/ivargr/graph_kmer_index). Relies on Numpy and some Cython to achieve fast mapping. Reads are read as bytes using Cython and kmers are extracted efficiently by using Numpy's convolve function. 

### Installation
Required Python 3.
```bash
pip install kmer_mapper
```

### Usage
Kmer mapper needs a two-line fasta. 
```bash
kmer_mapper map -i kmer_index.npz -f reads.fa -o results --n-threads 10
```
