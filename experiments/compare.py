import sys
sys.path.append('/home/vancemiller/github/bigchem/python')
import time, numpy as np, pandas as pd, bitstring, itertools
from rdkit import Chem
from gpu_tanimoto_popcount import GPUtanimoto
old_data = '/data/bigchem/data/example50K.h5'
new_data = '/data/bigchem/data/example50K-hex.h5'

def old_method(begin_query, end_query, begin_target, end_target):
    store = pd.HDFStore(old_data, )
    query = store.select('data', start=begin_query, stop=end_query)
    target = store.select('data', start=begin_target, stop=end_target)
    similarity = []

    # timing
    start_time = time.time()
    for v in query.values:
        similarity.append(tanimoto_matrix_multiplication(v, target.values))
    total_time = time.time() - start_time

    print "---------------------------------------"
    print "total time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % ((len(query) * len(target))/total_time)
    print "---------------------------------------"

    return similarity

def tanimoto_matrix_multiplication(vector, matrix):
    """
    Calculating pairwise Tanimoto distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    denom = np.square(np.linalg.norm(matrix, axis=1)) + np.square(np.linalg.norm(vector)) - dotted
    neighbors = np.divide(dotted, denom)
    return neighbors

def new_method(begin_query, end_query, begin_target, end_target):
    N = 16 # divide data into 16 chunks
    store = pd.HDFStore(new_data, )
    query = store.select('data', start=begin_query, stop=end_query)
    target = store.select('data', start=begin_target, stop=end_target)
    query_fp = np.empty((end_query - begin_query, N), dtype=np.uint64)
    query_fp = query['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))
    target_fp = np.empty((end_target - begin_target, N), dtype=np.uint64)
    target_fp = target['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))

    merged_query = np.reshape(list(itertools.chain.from_iterable(query_fp.values)), (end_query - begin_query, N))
    merged_target = np.reshape(list(itertools.chain.from_iterable(target_fp.values)), (end_target - begin_target, N))

    similarity = GPUtanimoto(merged_query, merged_target)
    return similarity

def hex2int64(x):
    return np.uint64(bitstring.BitArray(hex=x).uint)

def fphex2int64(x, length):
    # length of the integer 16 x 4bit from hex = 64
    y = map(''.join, zip(*[iter(x)] * length))
    return map(hex2int64, y)


if __name__ == "__main__":
    old_out = old_method(0, 100, 0, 100)
    new_out = new_method(0, 100, 0, 100)

    similarity_bound = 0.0000001

    exit()
    print "Comparing", len(old_out), "results"
    for i in range(len(old_out[0])):
        for j in range(len(old_out)):
            if (abs(old_out[i][j] - new_out[i * len(old_out[i]) + j][1]) > similarity_bound):
                print "comparison of index", (i, j), "differs by more than", similarity_bound


