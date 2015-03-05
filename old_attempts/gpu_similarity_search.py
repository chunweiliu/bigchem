import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void ssd(int *dest, int *a, int *b)
{
  const int i = threadIdx.x;
  // change the below calculation to use popcount implementation that Olex is sending
  dest[i] = (*a - b[i]) * (*a - b[i]);
}
""")

def GPUssd(a, b, cutoff=0, count):
    a = a.astype(numpy.int32);
    b = b.astype(numpy.int32);
    ssd = mod.get_function("ssd");
    output = [];
    for record in a:
        # check size of memory in GPU
        # if less than size of b, continue
        # else, break b into
        dest = numpy.zeros_like(b);
        ssd(drv.Out(dest), drv.In(a), drv.In(b), block=(400, 1, 1), grid=(1, 1));
        # format dest to be a tuple with record id and similarity metric
        # ...
        output.append(dest);

    # remove results less than cutoff
    #...
   # if cutoff != 0:
        # linear time remove items less than cutoff
    # if count is None:
    # continue?
    # if count < len(a) * len(b):
        # depending on count... linear time or sort

    return output;

if "__main__":
    a = numpy.random.randn(400);
    b = numpy.random.randn(400);
    dest = GPUssd(a, b);
    ans = sum(((a - b)*(a - b)) ** 0.5)
    print numpy.sum(dest) - ans  # difference of double and float

