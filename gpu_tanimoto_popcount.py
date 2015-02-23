import pycuda.autoinit;
import pycuda.driver as drv;
import numpy as np;
import math;

from pycuda.compiler import SourceModule;

mod = SourceModule("""
        __global__ void tanimoto_popcount(unsigned long long query, unsigned long long *target, int *out)
        {
            int a = 0, b = 0, c = 0;
            int idy = blockDim.y * blockIdx.y + threadIdx.y;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            a = __popcll(query);
            b = __popcll(target[idx]);
            c = __popcll(query & target[idx]);
            out[idx] = b;//((double) c) / ((double) a + b - c);
        }
""");

def GPUtanimoto(query, target, cutoff=0, count=None):
    query = query.astype(np.uint64);
    target = target.astype(np.uint64);

    tanimoto = mod.get_function("tanimoto_popcount");

    output = [];
    for q in query:
        q = np.arange(q, q+1);
        # check size of GPU memory
        dest = np.zeros((len(target)), np.int32);
        print query;
        print target;
        print dest;

        threads_per_block = 1024;
        blocks_per_mp = 4;

        tanimoto(drv.In(q), drv.In(target), drv.Out(dest), block=(64, 2, 1), grid=(1, 1));
        print dest;
        output.append(dest);
    return output;

if "__main__":
    target = np.arange(2048, 2056);
    dest = GPUtanimoto(np.arange(2048, 2050), target);
    print dest;

