import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle 
from scipy.stats import f_oneway

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)


for group in results.keys():
    dat = results[group]['dat']
    rat = results[group]['rat']
    print(group)
    print('DAT:', np.nanmean(dat))
    print('RAT:', np.nanmean(rat))
