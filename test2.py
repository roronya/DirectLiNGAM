from DirectLiNGAM import DirectLiNGAM, draw_causal_graph
import numpy as np
import pandas as pd

a = np.random.exponential(size=500)
b = np.random.exponential(size=500) + a
c = np.random.exponential(size=500) + a + b
d = np.random.exponential(size=500) + a + b + c
e = np.random.exponential(size=500) + a + b + c + d
data = np.array([a,b,c,d,e])
labels = ['a','b','c','d','e']
result = DirectLiNGAM(data, processes=1)
result_order = result[0]
result_matrix = result[1]
draw_causal_graph(result_order, result_matrix, labels, output_name='test2')
