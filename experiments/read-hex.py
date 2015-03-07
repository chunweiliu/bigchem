from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
import bitstring, time
import itertools

from gpu_tanimoto_popcount import GPUtanimoto;

def hex2int64(x):
    return np.uint64(bitstring.BitArray(hex=x).uint)

def fphex2int64(x):
    # length of the integer 16 x 4bit from hex = 64
    length=N;

    y = map(''.join, zip(*[iter(x)]*length))

    return map(hex2int64, y)


global N
global chunks

# size of the batch of data
chunks=10000

#1024 bitstring or 256 bit hex to give 16 int64
N = 16

# reading data
store = pd.HDFStore('/data/bigchem/data/example50K-hex.h5', )

workingSet = store.select('data', start=0, stop=chunks)
#print  workingSet.head(5)

fp_array = np.empty((chunks, N), dtype=np.uint64)
fp_array = workingSet['RDK5FPhex'].apply(lambda x: fphex2int64(x))


merged = np.reshape(list(itertools.chain.from_iterable(fp_array.values)), (chunks, N))
print(merged.shape)

GPUtanimoto(np.array([merged[0]]), merged);
