import numpy as np
import pandas as pd

import json
import pyarrow as pa
import lancedb

DIM_VALUE = 128

# class TextEmbeder:
#     def __init__(self, model_path: str = None):
#         pass

#         def encode(self, text: str) -> np.ndarray:
#             return np.

print(np.array([1, 2, 3]))
for x in np.array([1, 2, 3]):
    print(x)

print(np.zeros((2, 3)))

print(np.ones((2, 3)))

print('empty: ', np.empty((2, 3)), type(np.empty((2, 3))))

print(np.arange(3, 10, 2))

print(np.linspace(0, 1, 5), type(np.linspace(0, 1, 5)))

print(np.random.rand(2, 3))


print(np.random.randn(2, 3))