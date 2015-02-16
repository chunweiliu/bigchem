import numpy as np
import pandas as pd
import time
from rdkit import Chem


def TanimotoSimilarity(A, B):
    B = np.array(map(int, B))
    return np.dot(A, B) / \
        (np.absolute(np.dot(A, A)) + np.absolute(np.dot(B, B)) - np.dot(A, B))

if __name__ == "__main__":
    # Generate a query
    mol = Chem.MolFromSmiles("CCCN(N=C)C(C)C(=O)NO")
    fp_ref = Chem.RDKFingerprint(
        mol, maxPath=5, fpSize=1024, nBitsPerHash=2).ToBitString()
    a = np.array(map(int, fp_ref))

    # Read a subset of the entire data set
    store = pd.HDFStore(
        '/home/olexandr/github/gpu-htc/speedtest/gdb13.rand1M.h5', )
    workingSet = store.select('data', columns=['Smiles', 'RDK5'], start=0,
                              stop=50)
    #print workingSet.head(5)

    # Timing the performance
    start_time = time.time()
    workingSet['Tanimoto'] = workingSet['RDK5'].apply(
        lambda x: TanimotoSimilarity(a, x))  # Imporve the similarity function
    total_time = time.time() - start_time
    print "---------------------------------------"
    print "total time time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % (len(workingSet)/total_time)
    print "---------------------------------------"
