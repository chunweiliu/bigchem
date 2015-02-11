import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
mod = SourceModule("""
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

gpuPdist = mod.get_function("gpuPdist")

num_element = 1
num_feature = 1
a = numpy.random.randn(num_element, num_feature).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
gpuPdist(drv.Out(dest), drv.In(a), drv.In(num_element), drv.In(num_feature),
         block=(400, 1, 1), grid=(1, 1))

# ans = sum(((a - b)*(a - b)) ** 0.5)
# print numpy.sum(dest) - ans
# print dest-a*b
