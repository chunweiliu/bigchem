import sys
sys.path.append('/home/vancemiller/github/bigchem/python')
import time, numpy as np, pandas as pd, bitstring, itertools
from rdkit import Chem
from gpu_tanimoto_popcount import GPUtanimoto
data = './example50K-hex.h5'

def call_CUDA_popcount(begin_query, end_query, begin_target, end_target):
    N = 16 # divide data into 16 chunks
    store = pd.HDFStore(data, )
    query = store.select('data', start=begin_query, stop=end_query)
    target = store.select('data', start=begin_target, stop=end_target)

    if (len(query) < (end_query - begin_query) or len(target) < (end_target - begin_target)):
        print("Warning: did not read all requested entries.")

    print("converting...")
    query_fp = np.empty((end_query - begin_query, N), dtype=np.uint64)
    query_fp = query['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))
    target_fp = np.empty((end_target - begin_target, N), dtype=np.uint64)
    target_fp = target['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))
    print("reshaping...")
    query_list = list(itertools.chain.from_iterable(query_fp.values))
    target_list = list(itertools.chain.from_iterable(target_fp.values))
    merged_query = np.reshape(query_list, (end_query - begin_query, N))
    merged_target = np.reshape(target_list, (end_target - begin_target, N))
    print("computing similarity...")
    similarity = GPUtanimoto(merged_query, merged_target)

    return similarity

def hex2int64(x):
    return np.uint64(bitstring.BitArray(hex=x).uint)

def fphex2int64(x, length):
    # length of the integer 16 x 4bit from hex = 64
    y = map(''.join, zip(*[iter(x)] * length))
    return map(hex2int64, y)


if __name__ == "__main__":
    out = call_CUDA_popcount(0, 50000, 0, 50000)
    sys.exit(0)
#endif

