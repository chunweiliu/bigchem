import old_method, read_hex, time
chunks = 100
old_out = old_method.run('/data/bigchem/data/example50K.h5', chunks)
time.sleep(2)
new_out = read_hex.run('/data/bigchem/data/example50K-hex.h5', chunks)

similarity_bound = 0.0000001

exit()
print "Comparing", len(old_out), "results"
for i in range(len(old_out[0])):
    for j in range(len(old_out)):
        if (abs(old_out[i][j] - new_out[i * len(old_out[i]) + j][1]) > similarity_bound):
            print "comparison of index", (i, j), "differs by more than", similarity_bound

