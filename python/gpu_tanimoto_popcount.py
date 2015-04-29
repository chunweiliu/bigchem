from __future__ import print_function
import numpy as np
import time
import sys

from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.autoinit

"""
The CUDA kernel and subroutines for computing the tanimoto similarity.
Prepare this variable to be used as a function by calling:
funct = cuda_popcount.get_function("tanimoto_popcount")
The following module is written in C using CUDA extensions.
"""
cuda_popcount = SourceModule("""
  /**
   * Similarity function.
   * - This is a helper method that compares two molecules and returns their Tanimoto similarity.
   * @param query: One molecule represented as an array of 64-bit chunks of the molecule's bit signature.
   * @param target: Another molecule also represented as an array of 64-bit chunks of the molecule's bit signature.
   * @param data_length: The number of chunks in the array
   * @param cutoff: If the output of the comparison is below this value, the output will be set to zero.
   * @returns a float between 0 and 1 representing the Tanimoto similarity between query and target.
   */
  __device__ float similarity(unsigned long long *query,
                              unsigned long long *target, int data_len,
                              float cutoff) {
    int a = 0, b = 0, c = 0;
    float result;
    // iterate over the length of the query and target arrays
    for (int i = 0; i < data_len; i++) {
      // use CUDA specific popcount function to do the comparison
      a += __popcll(query[i]);
      b += __popcll(target[i]);
      c += __popcll(query[i] & target[i]);
    }
    // Handle the edge case where denominator is zero
    if (a + b == c) {
      result = 1.0f;
    }
    // else do the normal computation for Tanimoto similarity
    else {
      result = (float) c / (a + b - c);
    }
    // check if the result is less than the cutoff
    if (result < cutoff) {
      result = 0;
    }
    // finally, return the result
    return result;
  }

  /**
   * Tanimoto Popcount function.
   * - This function returns the pairwise Tanimoto similarity between query and targets.
   * @param query: An array of molecules. Each molecule is an array of 64-bit consecutive chunks
   *    of the molecule's bit signature.
   * @param query_len: The number of entries in the query array.
   * @param target: An array of molecules.
   * @param target_len: The number of entries in the target array.
   * @param data_len: The number of 64-bit consecutive chunks that make up each molecule's bit signature.
   * @param cutoff: If the output of a comparison is below this value, the output for that element will be set to zero.
   * @param out: A float array of size query_len * target_len. Output will be
   *    written to this array with queries going down the columns and targets across the rows.
   *    Expected output format (q = query, t = target)
   *        t1  t2  t3
   *    q1  .   .   .
   *    q2  .   .   .
   *    q3  .   .   .
   */
  __global__ void tanimoto_popcount(unsigned long long *query, int query_len,
                                    unsigned long long *target, int target_len,
                                    int data_len, float cutoff, float *out) {
    // Compute which indices in the output matrix we should operate on.
    // These parameters are set on kernel launch
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    // if we have a valid index, do the computation
    if (idy < query_len && idx < target_len) {
      // compute our location in the output array and do the comparison
      out[idy * target_len + idx] =
      similarity(&query[idy * data_len], &target[idx * data_len], data_len,
                 cutoff);
    }
  }
""")


def GPUtanimoto(query, target, cutoff=0, output_path="similarity_matrix.npy"):
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
    index of the first entry of the block (target, query) appended to their
    name.
    Default output path is "./similarity_matrix" and files will be written as
    similarity_matrix_[idx]_[idy].npy. (optional)
    """

    # Make sure that the inputs are properly formatted
    if len(query) == 0 or len(target) == 0:
        raise ValueError("Error, input must be of nonzero length.")

    # We need to properly format the input as uint64 matrices
    format_print("Converting inputs to np.uint64s")
    query = query.astype(np.uint64)
    target = target.astype(np.uint64)

    # Get a reference to our CUDA function
    tanimoto = cuda_popcount.get_function("tanimoto_popcount")

    # CUDA kernel size parameters
    query_size = len(query)
    target_size = len(target)
    threads_per_block = 32  # constant dependent on hardware

    # Loop, reducing memory size until we can fit the target_idxob on the GPU
    format_print("Attempting to execute on GPU. Looking for right memory size...")
    not_enough_memory = True
    blocks_written = 0
    while not_enough_memory:
        # Output array
        dest_in = np.zeros((target_size, query_size), dtype=np.float32)
        # Determine the block and grid sizes
        dx, mx = divmod(len(target), threads_per_block)
        dy, my = divmod(len(query), threads_per_block)
        bdim = (threads_per_block, threads_per_block, 1)
        gdim = ((dx + (mx > 0)), (dy + (my > 0)))
        # Call the CUDA
        try:
            target_idx = 0
            while target_idx < len(target):
                query_idx = 0
                format_print("Trying input of size {0} x {1}".format(query_size, target_size))
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
                    format_print("Success: done with chunk: {0}".format(blocks_written))
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
                    raise MemoryError("Unable to allocate memory. GPU is out of memory.")
                #endif
            #endif
        #endtry
    #endwhile
    format_print("Finished")
    return (target_size, query_size, blocks_written, output_path)

def format_print(text):
    """ Prints our message and the current time."""
    pattern = ': '
    print("STATUS " + time.asctime(time.localtime())[11:19] + pattern + text, file=sys.stderr)
