import pandas as pd
from rdkit import Chem, DataStructs
import bitstring


# reading data
store = pd.HDFStore('/data/bigchem/data/example50K-hex.h5', )

workingSet = store.select('data', start=0, stop=1000)
print  workingSet.head(5)


example = workingSet.iloc[0].RDK5FPhex
print(example)

# length 256 i.e 4bit x 256 = 1024
print(len(example))

#recover original bitstring
exampleBit = bitstring.BitArray(hex=example).bin

print(exampleBit)

# calculate fingerprint
mol = Chem.MolFromSmiles("CC1C(C)C(C(CC#N)C=C)C1C")
originalBit = Chem.RDKFingerprint(mol, maxPath=5, fpSize=1024, nBitsPerHash=2).ToBitString()

# verify that they are equal
print(exampleBit == originalBit)


#example of encoding to hex
def Hexifier(smi):
    return bitstring.BitArray(bin=Chem.RDKFingerprint(mol, maxPath=5, fpSize=1024).ToBitString()).hex


print(Hexifier(mol))

#verify that hex is identical
originalhex = Hexifier(mol)
print(example == originalhex)

