#!/usr/bin/python3
# Javier Ortiz de Artiñano Martínez-Larraz
# TFM MSc Data Science
import networkx as nx
import numpy as np
import json
import os

########################################
# NETWORKS
# Generates a set of scale-free networks with a given network size N
# and m new links of each new node.
def network_generator(N : int, m : int, n : int = 1, path = 'networks') -> None:
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in range(n):
        if os.path.exists(f"{path}/network_N_{N}_m_{m}_{i}.json"):
            continue
        graph = nx.barabasi_albert_graph(N, m)
        serialized_graph = json.dumps(graph, default=nx.node_link_data)
        with open(f"{path}/network_N_{N}_m_{m}_{i}.json", "w") as fh:
            fh.write(serialized_graph)
    return

# Generates an initial condition given a network size.
def IV_generator(N : int, path : str = "IVs") -> None:
    np.save(f"{path}/IV_{N}", np.random.rand(3 * N))
    return

if __name__ == '__main__':
    # Training set: Generate 15 networks for each size with m = 2, 3.
    for N in (50, 100, 150, 200, 250, 500):
        IV_generator(N)
        for m in (2,3):
            network_generator(N, m, 15, path = "networks/train")
    
    # Test set: 3 networks of each size with m = 2, 3
    for N in (50, 100, 150, 200, 250, 500):
        for m in (2,3):
            network_generator(N, m, 3, path = "networks/test")

    print("Done")