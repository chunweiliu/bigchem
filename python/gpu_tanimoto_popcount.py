import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
import time

from operator import itemgetter
from pycuda.compiler import SourceModule

#Define the CUDA kernel and subroutines
mod = SourceModule("""
  __device__ float similarity(unsigned long long *query,
                               unsigned long long *target, int data_len) {
    int a = 0, b = 0, c = 0;
    for (int i = 0; i < data_len; i++) {
      a += __popcll(query[i]);
      b += __popcll(target[i]);
      c += __popcll(query[i] & target[i]);
    }
    if (a + b == c) {
      return 1.0f;
    }
    else {
      return (float) c / (a + b - c);
    }
  }

  __global__ void tanimoto_popcount(unsigned long long *query, int query_len,
                                    unsigned long long *target, int target_len,
                                    int data_len, float *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idy < query_len && idx < target_len) {
      out[idy + idx * query_len] =
      similarity(&query[idy * data_len], &target[idx * data_len], data_len);
    }
  }
""")

def GPUtanimoto(query, target, cutoff=0, count=None):
    """
    This function computes the pairwise tanimoto similarity between the query and target. Query and target should be appropriately formatted as a matrix where each row is a molecule and each column is a 64 bit unsigned integer that represents a chunk of its bit sequence. Cutoff removes values from the output that are less than it and count specifies the number of items to return.
    """
    # We need to properly format the input as uint64 matrices
    query = query.astype(np.uint64)
    target = target.astype(np.uint64)
    tanimoto = mod.get_function("tanimoto_popcount")

    # CUDA kernel size parameters
    query_size = len(query)
    target_size = len(target)
    threads_per_block = 32  # constant dependent on hardware

    # List for gathering the output
    output_vectors = []
    array_index = 0

    not_enough_memory = True
    # Loop, reducing memory size until we can fit the job on the GPU
    while not_enough_memory:
        print "query size", query_size
        print "target size", target_size
        # Output array
        dest_in = np.zeros((target_size, query_size), dtype=np.float32)
        # Determine the block and grid sizes
        dx, mx = divmod(len(target), threads_per_block)
        dy, my = divmod(len(query), threads_per_block)
        bdim = (threads_per_block, threads_per_block, 1)
        gdim = ((dx + (mx > 0)), (dy + (my > 0)))
        # Call the CUDA
        start_time = time.time()
        try:
            j = 0
            while j < len(target):
                k = 0
                if (j + target_size > len(target)):
                    target_in = target[j:len(target)]
                else:
                    target_in = target[j:j + target_size]
                while k < len(query):
                    if (k + query_size > len(query)):
                        query_in = query[k:len(query)]
                    else:
                        query_in = query[k:k + query_size]

                    tanimoto(drv.In(query_in), np.int32(len(query_in)), drv.In(target_in), np.int32(len(target_in)), np.int32(len(query_in[0])), drv.Out(dest_in), block=bdim, grid=gdim)
                    not_enough_memory = False
                    print "done"
                    output_vectors.append(dest_in)
                    k = k + target_size
                #endwhile
                j = j + query_size
            #endwhile
        except pycuda._driver.MemoryError:
            print "out of memory"
            # We could not fit the job on the GPU.
            # Reduce memory requirements:
            if (target_size > 1):
                target_size = target_size / 2
            else:
                query_size = query_size / 2
                if (query_size == 0):
                    print "Unable to allocate memory"
                    sys.exit(1)
                #endif
            #endif
        #endtry
    #endwhile

    total_time = time.time() - start_time
    print "---------------------------------------"
    print "total time: %.3f seconds" % total_time
    print "Similarity speed %.3f Tanimoto/sec." % ((len(query)*len(target))/total_time)
    print "---------------------------------------"

    # gather the results
    results = []
    for out in output_vectors:
        for row in out:
            print "adding", array_index
            for entry in row:
                if entry >= cutoff:
                    results.append((array_index, entry))
                array_index = array_index + 1
            #endfor
        #endfor
    #endfor

    # Get the first count items
    if (count is not None):
        # sort on the similarity score field (item 1)
        results.sort(key=itemgetter(1))
        results = results[-count-1:-1]
    #endif

    return results

if __name__ == "__main__":

    query_range = range(2**0, 8)
    target_range = range(2**0, 4)

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
