#!/usr/bin/python3
# Javier Ortiz de Artiñano Martínez-Larraz
# TFM MSc Data Science

import numpy as np
import pandas as pd
import networkx as nx
from time import perf_counter
from utils import retrieve_network, synchronization_measure, interpolation_with_poincare_section, cross_correlation_synchronization

solutions_path = "simulations/train"

# Simulation parameters------------------------
t_final = 3000
ppt = 1
tiempo = np.linspace(0, t_final, int(t_final * ppt)) #ppt valores por segundo
t_span = (0, t_final)

g_list = np.linspace(0, 0.25, 20)


# Network
network_path = 'networks/train'

# Desired fields: simulación (key), N, m, i, g, neuron, median number of peaks, degree
neuron_df = pd.DataFrame(columns= [
    "key", "N", "m", "i", "Neuron", "Number of peaks", "Degree"
])
for N in (50,100, 150, 250, 500):
    for m in (2,3):
        for i in range(5):
            graph = retrieve_network(N, m, i, path = network_path)
            laplacianMatrix = nx.laplacian_matrix(graph).todense()
            for g in g_list:
                key = f'N_{N}_m_{m}_{i}_g_{round(g,4)}'
                print(f"Processing {key}...", end = " ")
                start = perf_counter()
                peaks_df = pd.read_csv(f"{solutions_path}/peaks/{key}.csv")
                peaks_df.loc[:, "key"] = key
                neuron_df = pd.concat([neuron_df, peaks_df.groupby(["key", "Neuron"]).median("Number of peaks").reset_index()[["key", "Neuron", "Number of peaks"]]])
                neuron_df.loc[neuron_df["key"] == key, "N"] = N
                neuron_df.loc[neuron_df["key"] == key, "m"] = m
                neuron_df.loc[neuron_df["key"] == key, "i"] = i
                neuron_df.loc[neuron_df["key"] == key, "g"] = round(g, 4)
                neuron_df.loc[neuron_df["key"] == key, "Degree"] = pd.Series(np.diagonal(laplacianMatrix))
                end = perf_counter()
                print(f"Done! It took {round(end - start, 4)} s")
neuron_df.reset_index(drop = True)
print("Done! Measuring synchronization...")

for key in neuron_df["key"].unique():
    print(f"Processing {key}...", end = " ")
    start = perf_counter()
    x = np.load(f"{solutions_path}/x_{key}.npy")
    neuron_df.loc[neuron_df["key"] == key, "Amplitude sync"] = cross_correlation_synchronization(x)
    valleys_df = pd.read_csv(f"{solutions_path}/valleys/{key}.csv")
    neuron_df.loc[neuron_df["key"] == key, "Phase sync"] = synchronization_measure(interpolation_with_poincare_section(x, valleys_df), method = "kura")
    end = perf_counter()
    print(f"Done! It took {round(end - start, 4)} s")
neuron_df.to_csv(f"{solutions_path}/neuron_df.csv")
print("Done!")

