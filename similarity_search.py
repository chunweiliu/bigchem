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


if __name__ == "__main__":
    # Generate a query
    mol = Chem.MolFromSmiles("CCCN(N=C)C(C)C(=O)NO")
    fp_ref = Chem.RDKFingerprint(
        mol, maxPath=5, fpSize=1024, nBitsPerHash=2).ToBitString()
    a = np.array(map(int, fp_ref))

    # Read a subset of the entire data set
    store = pd.HDFStore('/data/bigchem/data/example50K.h5', )
    workingSet = store.select('data', start=0, stop=10000)
    #print workingSet.head(5)

    # Timing the performance
    start_time = time.time()

    #  old line by line
    #workingSet['Tanimoto'] = workingSet['RDK5'].apply(
    #    lambda x: TanimotoSimilarity(a, x))  # Imporve the similarity function

    #new way
    similarity = tanimoto_matrix_multiplication(a, workingSet.values)

    total_time = time.time() - start_time
    print "---------------------------------------"
    print "total time time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % (len(workingSet)/total_time)
    print "---------------------------------------"
