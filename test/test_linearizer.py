import numpy as np
import unittest

from tinygrad.lazy import Device
from tinygrad.ops import GlobalCounters, Compiled
from tinygrad.tensor import Tensor

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled supports cache")
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    GlobalCounters.cache = []
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()
    rawbufs = GlobalCounters.cache[0][1]
    GlobalCounters.cache = None
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.realized, b.lazydata.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy())
  def test_tc(self):
    res = Tensor.ones(8,8).contiguous().matmul(Tensor.ones(8,8))
    print(res.lazydata.realize().toCPU())
  def test_lo(self):
    res = Tensor.rand(8,8).matmul(Tensor.rand(8,8))
    print(res.lazydata.realize().toCPU())
  def test_sm(self):
    res = Tensor.ones(8,8).contiguous()+Tensor.ones(8,8)
    print(res.lazydata.realize().toCPU())

if __name__ == '__main__':
  unittest.main()
