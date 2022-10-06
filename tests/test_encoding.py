import numpy as np
from bionumpy.encodings import ACTGEncoding
from kmer_mapper.encodings import ACTGTwoBitEncoding
from npstructures import RaggedArray
from bionumpy.encodings.alphabet_encoding import ACTGArray
from bionumpy.sequences import ASCIIText

# Only tests for checking the interface of bionumpy


def test():
    n = 100
    alphabet = np.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=np.uint8)
    sequences = np.random.choice(alphabet, n)

    sequences = np.array([65, 67], dtype=np.uint8)
    encoded = sequences.view(ASCIIText)
    print(encoded.ravel() == "AC")


if __name__ == "__main__":
    test()




