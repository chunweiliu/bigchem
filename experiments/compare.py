import sys
sys.path.append('/home/vancemiller/github/bigchem/python')
import time, numpy as np, pandas as pd, bitstring, itertools
from rdkit import Chem
from gpu_tanimoto_popcount import GPUtanimoto
old_data = '/data/bigchem/data/example1M.h5'
new_data = '/data/bigchem/data/example1M-hex.h5'

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
#    old_out = old_method(0, 500, 0, 500)
    new_out = new_method(0, 100000, 0, 100000)
    similarity_bound = 0.0001
    sys.exit(0)
    print "Comparing", len(old_out), "results"
    for i in range(len(old_out)):
        errors_exist = False
        count = 0
        if (abs(old_out[i] - new_out[i][1]) > similarity_bound):
            count = count + 1
        #endif
    #endfor
    if count > 0:
        print count, "errors"
    #endif
#endif

