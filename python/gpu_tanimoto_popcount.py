from __future__ import print_function
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.autoinit

import numpy as np
import time, sys

"""
The CUDA kernel and subroutines for computing the tanimoto similarity.
Prepare this variable to be used as a function by calling:
funct = cuda_popcount.get_function("tanimoto_popcount")
"""
cuda_popcount = SourceModule("""
  __device__ float similarity(unsigned long long *query,
                              unsigned long long *target, int data_len,
                              float cutoff) {
    int a = 0, b = 0, c = 0;
    float result;
    for (int i = 0; i < data_len; i++) {
      a += __popcll(query[i]);
      b += __popcll(target[i]);
      c += __popcll(query[i] & target[i]);
    }
    if (a + b == c) {
      return 1.0f;
    }
    else {
      result = (float) c / (a + b - c);
      if (result < cutoff) {
        result = 0;
      }
    }
    return result;
  }

  __global__ void tanimoto_popcount(unsigned long long *query, int query_len,
                                    unsigned long long *target, int target_len,
                                    int data_len, float cutoff, float *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idy < query_len && idx < target_len) {
      out[idy + idx * query_len] =
      similarity(&query[idy * data_len], &target[idx * data_len], data_len, cutoff);
    }
  }
""")


def GPUtanimoto(query, target, cutoff=0, output_path="similarity_matrix"):
    """
    Returns the pairwise similarity between query and target as a list of
    tuples. Each tuple is (index, similarity) where index is the location
    in the 2d matrix: query X target.

    == PARAMETERS ==
    @param query: A matrix of np.uint64. Rows represent individual molecules.
    Columns represent 64-bit chunks of their bit string representations.

    @param target: A matrix of np.uint64. Rows represent individual molecules.
    Columns represent 64-bit chunks of their bit string representations.

    @param cutoff: A np.float32 that specifies a value below which individual
    similarity computation results should be set to zero. (optional)

    @param output_path: Output will be written to .npy files under this path
    (includes file name). If the input is too large to fit in memory, multiple
    files will be created. If multiple files are created, they will have the
    index of the first entry of the block (target, query) appended to their name.
    Default output path is "./similarity_matrix" and files will be written as
    similarity_matrix_0_0.npy. (optional)
    """
    # Make sure that the inputs are properly formatted
    if len(query) == 0 or len(target) == 0:
        return []
    # We need to properly format the input as uint64 matrices
    print("Convert inputs to np.uint64s", file=sys.stderr)
    query = query.astype(np.uint64)
    target = target.astype(np.uint64)
    tanimoto = cuda_popcount.get_function("tanimoto_popcount")

    # CUDA kernel size parameters
    query_size = len(query)
    target_size = len(target)
    threads_per_block = 32  # constant dependent on hardware

    # Loop, reducing memory size until we can fit the target_idxob on the GPU
    not_enough_memory = True
    blocks_written = 0
    print("Attempting to execute on GPU. Looking for right memory size...", file=sys.stderr)
    while not_enough_memory:
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
            target_idx = 0
            while target_idx < len(target):
                query_idx = 0
                print("Trying input of size ", query_size, "x", target_size, file=sys.stderr)
                if (target_idx + target_size > len(target)):
                    target_in = target[target_idx:len(target)]
                else:
                    target_in = target[target_idx:target_idx + target_size]
                while query_idx < len(query):
                    if (query_idx + query_size > len(query)):
                        query_in = query[query_idx:len(query)]
                    else:
                        query_in = query[query_idx:query_idx + query_size]

                    tanimoto(drv.In(query_in), np.int32(len(query_in)),
                             drv.In(target_in), np.int32(len(target_in)),
                             np.int32(len(query_in[0])), np.float32(cutoff), drv.Out(dest_in),
                             block=bdim, grid=gdim)
                    print("Success: done with chunk: ", blocks_written, file=sys.stderr)
                    not_enough_memory = False
                    np.save(output_path + "_" + str(target_idx) + "_" + str(query_idx), dest_in)
                    blocks_written += 1
                    query_idx = query_idx + target_size
                #endwhile
                target_idx = target_idx + query_size
            #endwhile
        except pycuda._driver.MemoryError:
            # We could not fit the target_idxob on the GPU.
            # Reduce memory requirements:
            if (target_size > 1):
                target_size = target_size / 2
            else:
                query_size = query_size / 2
                if (query_size == 0):
                    print("Unable to allocate memory", file=sys.stderr)
                    sys.exit(1)
                #endif
            #endif
        #endtry
    #endwhile

    total_time = time.time() - start_time
    print("new_time %.3f" % total_time)
    print("new_speed %.3f" % ((len(query)*len(target))/total_time))

    return (target_size, query_size, blocks_written, output_path)

