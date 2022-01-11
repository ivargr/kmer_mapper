import logging
logging.basicConfig(level=logging.INFO)
import sys
import random

entries = []

with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            logging.info("%d lines processed" % i)

        if line.startswith(">"):
            entries.append([line, None])
        else:
            entries[-1][1] = line.strip()


logging.info("shuffling")
random.shuffle(entries)
logging.info("Done shuffling")


for entry in entries:
    print(''.join(entry))