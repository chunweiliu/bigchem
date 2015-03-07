import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
import time

from operator import itemgetter
from pycuda.compiler import SourceModule

mod = SourceModule("""
  __device__ double similarity(unsigned long long *query,
                               unsigned long long *target, int data_len) {
    int a = 0, b = 0, c = 0;
    for (int i = 0; i < data_len; i++) {
      a += __popcll(query[i]);
      b += __popcll(target[i]);
      c += __popcll(query[i] & target[i]);
    }
    /* Need to handle edge cases
    if (a + b == c) {
      return 1.0; // ask about this
    }
    else*/ {
      return (double) c / (a + b - c);
    }
  }

  __global__ void tanimoto_popcount(unsigned long long *query, int query_len,
                                    unsigned long long *target, int target_len,
                                    int data_len, double *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < target_len && idy < query_len) {
      out[idx + idy * target_len] =
      similarity(&query[idy * data_len], &target[idx * data_len], data_len);
    }
  }
""")


def GPUtanimoto(query, target, cutoff=0, count=None):
    query = query.astype(np.uint64)
    target = target.astype(np.uint64)
    tanimoto = mod.get_function("tanimoto_popcount")

    # TODO check size of GPU memory
    # Output array
    dest = np.zeros((len(query), len(target)), np.float64)
    # Determine the block and grid sizes
    threads_per_block = 32  # constant dependent on hardware
    dx, mx = divmod(len(target), threads_per_block)
    dy, my = divmod(len(query), threads_per_block)
    bdim = (threads_per_block, threads_per_block, 1)
    gdim = ((dx + (mx > 0)), (dy + (my > 0)))
    # Call the CUDA
    start_time = time.time()
    tanimoto(drv.In(query), np.int32(len(query)), drv.In(target), np.int32(len(target)), np.int32(len(query[0])), drv.Out(dest), block=bdim, grid=gdim)

    total_time = time.time() - start_time
    print "---------------------------------------"
    print "total time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % ((len(query)*len(target))/total_time)
    print "---------------------------------------"

    # Remove elements less than the cutoff
    data_subset = []
    array_index = 0
    for target_score in dest:
        for query_score in target_score:
            if (query_score >= cutoff):
                # Append the tuple (original index in array, similarity score)
                # to the list
                data_subset.append((array_index, query_score))
            array_index += 1
    # Get the first count items
    if (count is not None):
        # sort on the similarity score field (item 1)
        data_subset.sort(key=itemgetter(1))
        data_subset = data_subset[-count-1:-1]
    return data_subset

if __name__ == "__main__":

    query_range = range(2**0, 2**3)
    target_range = range(2**0, 2**3)

    query = np.array([[x for x in range(y, y + 16)] for y in query_range],
                     np.uint64)
    target = np.array([[x for x in range(y, y + 16)] for y in target_range],
                      np.uint64)

    # query = np.array([[0xFFF], [0x111], [0x222]]);
    # target = query;

    print("Input query:\n" + str(query))
    print("Number of queries: " + str(len(query)))
    print("Target data:\n" + str(target))
    print("Number of targets: " + str(len(target)))
    print("Data length: " + str(len(query[0])))

    output = GPUtanimoto(query, target)
    print("Output:\n" + str(output))
