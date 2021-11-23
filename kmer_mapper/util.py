import numpy as np
#import numpy_indexed as npi

def read_fasta(file_name):
    i = 0
    with open(file_name, "rb") as f:
        for line in f:
            if line[0] != 62:
                i += 1
                yield line



def remap_array(array, from_values, to_values):
    #from_values = np.array([0, 1, 2, 3])
    #to_values = np.array([100, 200, 300, 400])
    index = np.digitize(array.ravel(), from_values, right=True)
    return to_values[index].reshape(array.shape)


def remap_array2(array):
    return npi.remap(array, np.array([0, 1, 2, 3]), np.array([100, 200, 300, 400]))
