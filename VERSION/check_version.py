# test cntk.py
import sys
import numpy as np
import cntk as C

py_ver = sys.version
cntk_ver = C.__version__

print("Python: " + str(py_ver))
print("CNTK: " + str(cntk_ver))