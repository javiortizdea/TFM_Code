#!/usr/bin/python3
# Javier Ortiz de Artiñano Martínez-Larraz
# TFM MSc Data Science

import numpy as np
from time import perf_counter
import os
from utils import get_valleys_df, get_peaks_df

solutions_path = "simulations/train"

# Simulation parameters------------------------
t_final = 3000
ppt = 1
tiempo = np.linspace(0, t_final, int(t_final * ppt)) #ppt valores por segundo
t_span = (0, t_final)

# g_list = np.concatenate([[0], np.logspace(-2,0,10)])
g_list = np.linspace(0, 0.8, 20)
g_list = np.linspace(0, 0.25, 20)


# Network
network_path = 'networks/train'
valleys_dic = {}
peaks_dic = {}

if not os.path.exists(f'{solutions_path}/valleys'):
    os.mkdir(f'{solutions_path}/valleys')
if not os.path.exists(f'{solutions_path}/peaks'):
    os.mkdir(f'{solutions_path}/peaks')

for N in (50, 100, 150, 250, 500):
    for m in (2,3):
        for i in range(5):
            for g in g_list:
                key = f'N_{N}_m_{m}_{i}_g_{round(g,4)}'
                print(f"Clustering the peaks and valleys of {key}...", end = " ")
                start = perf_counter()
                x = np.load(f"{solutions_path}/x_{key}.npy")
                valleys_df = get_valleys_df(x, tiempo, only_manual=True, manual_threshold= -1.0)
                valleys_df.to_csv(f"{solutions_path}/valleys/{key}.csv")
                peaks_df = get_peaks_df(x, tiempo, valleys_df)
                peaks_df.to_csv(f"{solutions_path}/peaks/{key}.csv")
                end1 = perf_counter()
                print(f"It took {round(end1 - start, 2)} s")
print("Done!")