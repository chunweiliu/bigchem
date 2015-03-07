import numpy as np
import pandas as pd
import time
from rdkit import Chem


def TanimotoSimilarity(A, B):
    B = np.array(map(int, B))
    return np.dot(A, B) / \
        (np.absolute(np.dot(A, A)) + np.absolute(np.dot(B, B)) - np.dot(A, B))


def tanimoto_matrix_multiplication(vector, matrix):
    """
    Calculating pairwise Tanimoto distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    denom = np.square(np.linalg.norm(matrix, axis=1)) + np.square(np.linalg.norm(vector)) - dotted
    neighbors = np.divide(dotted, denom)
    return neighbors


def run(path_to_data, chunks):
    # Generate a query

    # Read a subset of the entire data set
    store = pd.HDFStore(path_to_data, )
    workingSet = store.select('data', start=0, stop=chunks)
    #print workingSet.head(5)

    # Timing the performance
    start_time = time.time()

    #  old line by line
    #workingSet['Tanimoto'] = workingSet['RDK5'].apply(
    #    lambda x: TanimotoSimilarity(a, x))  # Imporve the similarity function

    #new way
    similarity = [];
    for v in workingSet.values:
        similarity.append(tanimoto_matrix_multiplication(v, workingSet.values))

    total_time = time.time() - start_time
    print "---------------------------------------"
    print "total time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % (len(workingSet)/total_time)
    print "---------------------------------------"
    return similarity

if __name__ == "__main__":
    print run('/data/bigchem/data/example50K.h5', 10000);
