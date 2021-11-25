# Kmer Mapper
Fast numpy package for mapping kmers from a fasta file to a Kmer Index. Relies on numpy and some Cython to achieve fast mapping. Reads are read as bytes using Cython and kmers are extracted efficiently by using Numpy's convolve function. 

### Installation
```bash
pip install kmer_mapper
```

### Usage
```bash
kmer_mapper map -i kmer_index.npz -f reads.fa -o results --n-threads 10
```

