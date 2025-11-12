# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:59:30 2025

@author: Termohidraulica
"""

import os
import pickle
files = [f for f in os.listdir('.') if 'pkl' in f]
workspaces = {}
for file in files:
    with open(file,
              'rb') as f:
        workspaces[file] = pickle.load(f)
        

files

for k, v in workspaces.items():
    print(k, v['results']['irreducible_error'])

workspaces['output_chf_1_tube_5inputs_Parallel_12hs.pkl']['epsilon']

workspaces['output_chf_1_tube_5inputs_Parallel_kraskov.pkl']['results']['irreducible_error']

workspaces['output_chf_1_tube_5inputs_Parallel_kraskov.pkl']['results']['irreducible_error']

