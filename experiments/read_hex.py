from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
import bitstring, time
import itertools, sys

sys.path.append('/home/vancemiller/github/bigchem');
from gpu_tanimoto_popcount import GPUtanimoto;

def hex2int64(x):
    return np.uint64(bitstring.BitArray(hex=x).uint)

def fphex2int64(x, length):
    # length of the integer 16 x 4bit from hex = 64

    y = map(''.join, zip(*[iter(x)]*length))

    return map(hex2int64, y)



def run(path_to_data, chunks):
#1024 bitstring or 256 bit hex to give 16 int64
    N = 16

# reading data
    store = pd.HDFStore(path_to_data, )

    workingSet = store.select('data', start=0, stop=chunks)
#print  workingSet.head(5)

    fp_array = np.empty((chunks, N), dtype=np.uint64)
    fp_array = workingSet['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))


    merged = np.reshape(list(itertools.chain.from_iterable(fp_array.values)), (chunks, N))
    print(merged.shape)

    similarity = GPUtanimoto(merged, merged);
    return similarity;

if __name__ == "__main__":
    print(run('/data/bigchem/data/example50k.h5', 10000));
