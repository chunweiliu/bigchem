import pycuda.autoinit;
import pycuda.driver as drv;
import numpy as np;
import math;

from pycuda.compiler import SourceModule;

mod = SourceModule("""
        __global__ void tanimoto_popcount(unsigned long long *query, int query_len, unsigned long long *target, int target_len, int length_of_data, double *out)
        {
            int a = 0, b = 0, c = 0;
            int idy = (blockDim.y * blockIdx.y + threadIdx.y);
            int idx = (blockDim.x * blockIdx.x + threadIdx.x);
            if (idy < query_len && idx < target_len) {
                for (int i = 0; i < length_of_data; i++) {
                    a += __popcll(query[idy]);
                    b += __popcll(target[idx]);
                    c += __popcll(query[idy] & target[idx]);
                }
                out[idx + idy * blockDim.x] = ((double) c) / ((double) a + b - c);
            }
        }
""");

def GPUtanimoto(query, target, cutoff=0, count=None):
    query = query.astype(np.uint64);
    target = target.astype(np.uint64);

    tanimoto = mod.get_function("tanimoto_popcount");

    output = [];
    # check size of GPU memory
    dest = np.zeros((len(query), len(target)), np.float64);
    print query;
    print target;
    print dest;

    threads_per_block = 32; # dependent on hardware

    dx, mx = divmod(len(target), threads_per_block);
    dy, my = divmod(len(query), threads_per_block);

    bdim = (threads_per_block, threads_per_block, 1);
    gdim = ((dx + (mx > 0)), (dy + (my > 0)));

    print bdim;
    print gdim;

    tanimoto(drv.In(query), np.int32(len(query)), drv.In(target), np.int32(len(target)), np.int32(len(query[0])), drv.Out(dest), block=bdim, grid=gdim);
    return dest;

if "__main__":
    query = np.array([[x for x in range(y, y + 6)] for y in range(2048, 2050)], np.uint64);

    target = np.array([[x for x in range(y, y + 6)] for y in range(2048, 2050)], np.uint64);

    dest = GPUtanimoto(query, target);
    print dest;

