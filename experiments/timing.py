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

    for t in target.values:
        for q in query.values:
            similarity.append(TanimotoSimilarity(t, q))
    total_time = time.time() - start_time

    print "old_time %.3f" % total_time
    print "old_speed %.3f" % ((len(query) * len(target))/total_time)
    store.close()
    return similarity

def tanimoto_matrix_multiplication(vector, matrix):
    """
    Calculating pairwise Tanimoto distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    denom = np.square(np.linalg.norm(matrix, axis=1)) + np.square(np.linalg.norm(vector)) - dotted
    neighbors = np.divide(dotted, denom)
    return neighbors

def TanimotoSimilarity(A, B):
    A = np.array(map(float, A))
    B = np.array(map(float, B))
    return np.dot(A, B)/(np.absolute(np.dot(A, A)) + np.absolute(np.dot(B, B)) - np.dot(A, B))

def new_method(begin_query, end_query, begin_target, end_target):
    N = 16 # divide data into 16 chunks
    store = pd.HDFStore(new_data, )
    query = store.select('data', start=begin_query, stop=end_query)
    target = store.select('data', start=begin_target, stop=end_target)

    if (len(query) < (end_query - begin_query) or len(target) < (end_target - begin_target)):
        print("Warning: did not read all requested entries.")

    query_fp = np.empty((end_query - begin_query, N), dtype=np.uint64)
    query_fp = query['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))
    target_fp = np.empty((end_target - begin_target, N), dtype=np.uint64)
    target_fp = target['RDK5FPhex'].apply(lambda x: fphex2int64(x, N))

    query_list = list(itertools.chain.from_iterable(query_fp.values))
    target_list = list(itertools.chain.from_iterable(target_fp.values))
    merged_query = np.reshape(query_list, (end_query - begin_query, N))
    merged_target = np.reshape(target_list, (end_target - begin_target, N))

    similarity = GPUtanimoto(merged_query, merged_target)
    store.close()
    return similarity

def hex2int64(x):
    return np.uint64(bitstring.BitArray(hex=x).uint)

def fphex2int64(x, length):
    # length of the integer 16 x 4bit from hex = 64
    y = map(''.join, zip(*[iter(x)] * length))
    return map(hex2int64, y)


if __name__ == "__main__":
    for size in range(5000):
        print 'size', size
        old_out = old_method(0, size, 0, size)
        new_out = new_method(0, size, 0, size)
    #endfor
#endif

