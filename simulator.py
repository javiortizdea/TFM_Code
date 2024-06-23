#!/usr/bin/python3
# Javier Ortiz de Artiñano Martínez-Larraz
# TFM MSc Data Science

import networkx as nx
import numpy as np
from time import perf_counter
from utils import retrieve_network, simulation_HR
import os

try:
    os.mkdir("solutions")
    os.mkdir("solutions/train")
except:
    next

solutions_path = "simulations/train"
# Simulation parameters------------------------
t_final = 3000
ppt = 1
tiempo = np.linspace(0, t_final, int(t_final * ppt)) #ppt valores por segundo
t_span = (0, t_final)

g_list = np.linspace(0, 0.25, 20)

# Network
path = 'networks/train'

# Generate the simulations of the HR systems. It will take a long time.
for N in (50, 100, 150, 250, 500):
     condicionInicial = np.load(f"IVs/IV_{N}.npy")
     for m in (2,3):
          for i in range(5):
               graph = retrieve_network(N, m, i, path=path)
               laplacianMatrix = nx.laplacian_matrix(graph).todense()
               for g in g_list:
                    if os.path.exists(f"{solutions_path}/x_N_{N}_m_{m}_{i}_g_{round(g, 4)}.npy"):
                         print(f"Simulation x_N_{N}_m_{m}_{i}_g_{round(g, 4)} already exists")
                         continue
                    print(f"Simulating for N = {N}, m = {m}, g = {round(g,4)}...", end = " ")
                    start = perf_counter()
                    x, y, z = simulation_HR(N = N, t_final=t_final, 
                                            ppt = ppt, 
                                            laplacianMatrix=laplacianMatrix, 
                                            condicionInicial= condicionInicial, 
                                            g = g, method = "RK45", 
                                            atol = 1e-10, rtol = 1e-10)
                    end = perf_counter()
                    print(f"It took {end - start} s. Saving solutions...", end = " ")
                    np.save(f"{solutions_path}/x_N_{N}_m_{m}_{i}_g_{round(g, 4)}", x)
                    end2 = perf_counter()
                    print(f"Done! It took {end2 - end} s")