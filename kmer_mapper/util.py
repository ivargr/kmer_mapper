import numpy as np
#import numpy_indexed as npi
import resource
import logging

def read_fasta(file_name):
    i = 0
    with open(file_name, "rb") as f:
        for line in f:
            if line[0] != 62:
                i += 1
                yield line



def remap_array(array, from_values, to_values):
    index = np.digitize(array.ravel(), from_values, right=True)
    return to_values[index].reshape(array.shape)



def log_memory_usage_now(logplace=""):
    memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000
    logging.info("Memory usage (%s): %.4f GB" % (logplace, memory))
