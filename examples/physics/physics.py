import os
import sys
path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(path)

from DirectLiNGAM import DirectLiNGAM, draw_causal_graph
import numpy as np
import pandas as pd

data = pd.read_csv('res.txt', header=None, delim_whitespace=True).values.T
labels = ['theta1', 'theta2', 'omega1', 'omega2']
result = DirectLiNGAM(data, processes=16)
result_order = result[0]
result_matrix = result[1]
draw_causal_graph(result_order, result_matrix, labels, output_name='physics')
