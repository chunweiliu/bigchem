import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
mod = SourceModule("""
/**
 * GPU Manhattan distance calculation from Chang_etal_SNPD2009
 */
__global__ void
gpuManhattan(float *out, float *in, int n, int m){
    __shared__ float Xs[16][16];
    __shared__ float Ys[16][16];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int xBegin = bx * 16 * m;
    int yBegin = by * 16 * m;
    int yEnd = yBegin + m - 1, y, x, k, o;
    int x, y, k, o;
    float s = 0.0;

    for (y = yBegin, x = xBegin; y <= yEnd; y += 16, x += 16){
        Ys[ty][tx] = in[y + ty*m + tx];
         Xs[tx][ty] = in[x + ty*m + tx];
         //*** note the transpose of Xs
         __syncthreads();

         for(k = 0; k < 16; k++){
             s += fabs(Ys[ty][k] - Xs[k][tx];
        }
        __syncthreads();
    }
    o = by * 16 * n + ty * n + bx * 16 + tx;
    out[o] = s;
}
""")

gpuPdist = mod.get_function("gpuPdist")

num_element = 10
num_feature = 10

a = numpy.random.randn(num_element, num_feature).astype(numpy.float32)

dest = numpy.zeros_like(a)
gpuManhattan(drv.Out(dest), drv.In(a), numpy.int32(num_element), numpy.int32(num_feature),
         block=(1, 1, 1), grid=(1, 1))

print dest
