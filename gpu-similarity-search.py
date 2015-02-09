import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void ssd(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = sqrt((a[i] - b[i]) * (a[i] - b[i]));
}
""")

ssd = mod.get_function("ssd")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
ssd(drv.Out(dest), drv.In(a), drv.In(b),
    block=(400, 1, 1), grid=(1, 1))

ans = sum(((a - b)*(a - b)) ** 0.5)
print numpy.sum(dest) - ans  # difference of double and float
# print dest-a*b
