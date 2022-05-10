import unittest
import numpy as np
import os
import subprocess as sp
import sys

def check_two(cmd, parser=None, checker=None):
    jtorch_out = sp.getoutput(cmd)
    print("=========JTORCH OUT==========")
    print(jtorch_out)
    torch_out = sp.getoutput("PYTHONPATH= "+cmd)
    print("=========TORCH OUT==========")
    print(torch_out)
    if parser:
        torch_out = parser(torch_out)
        jtorch_out = parser(jtorch_out)
    if checker:
        checker(torch_out, jtorch_out)
    else:
        assert torch_out == jtorch_out
    return jtorch_out, torch_out

jtorch_path = os.path.join(os.path.dirname(__file__), "..")
class TestTutorial(unittest.TestCase):
    def test_auto_grad1(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad1.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))

if __name__ == "__main__":
    unittest.main()