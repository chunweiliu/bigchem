import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
mod = SourceModule("""
/**
 * GPU Pairwise distance calculation from Chang_etal_CBB2008_634-017
 */
__global__ void
gpuPdist(float *out, float *in, int n, int m){
__shared__ float Ys[16][16];
__shared__ float Xs[16][16];
int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;
int yBegin = by * 16 * m;
int xBegin = bx * 16 * m;
int yEnd = yBegin + m - 1, y, x, k, o;
float tmp, s = 0;

for(y=yBegin,x=xBegin; y<=yEnd; y+=16,x+=16){
  Ys[ty][tx] = in[y + ty*m + tx];
  Xs[tx][ty] = in[x + ty*m + tx];
  //*** note the transpose of Xs
  __syncthreads();

  for(k=0;k<16;k++){
    tmp = Ys[ty][k] - Xs[k][tx];
    s += tmp*tmp;
  }
  __syncthreads();
}

o = by*16*n + ty*n + bx*16 + tx;
out[o] = sqrtf(s);
}
""")

def GPUpairwise(a):
    gpuPdist = mod.get_function("gpuPdist");
    a = a.astype(numpy.float32);
    dest = numpy.zeros_like(a);
    num_elements = numpy.int32(a.shape[0]);
    num_features = numpy.int32(a.shape[1]);
    gpuPdist(drv.Out(dest), drv.In(a), num_elements, num_features, block=(1,1,1), grid=(1,1));
    return dest;

if "__main__":
    a = numpy.random.randn(10, 10);
    out = GPUpairwise(a);
    print out;
