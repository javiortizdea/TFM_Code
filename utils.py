#!/usr/bin/python3
import numpy as np
import scipy.integrate as scint
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
import json

def HR_model(t, initial_state, G, g, a = 1.0, b = 3.0, c = 1.0, d = 5.0, s = 4.0, r = 0.006, x_r = -1.56, I_bias = 2.9):
    # The system state is passed as a 1D array in which every ordered sequence of 3 elements defines
    # the x, y and z values of a neuron.
    x = initial_state[0::3] # all the x values
    y = initial_state[1::3] # all the y values
    z = initial_state[2::3] # all the z values
    N = len(x)
    
    electric_coupling_differences = np.sum(G * x, axis = 1)
    
    
    dX = np.empty(3*N)
    
    
    x2 = np.power(x,2)
    dX[0::3] = y - a * np.power(x, 3) + b * x2 - z + I_bias - g * electric_coupling_differences #dx/dt
    dX[1::3] = c - d * x2 - y #dy/dt
    dX[2::3] = r * (s * (x - x_r) - z) #dz/dt
    
    return dX

def simulation_HR(N = 25, t_final = 3000,
                  ppt = 5,
                  laplacianMatrix = None,
                  g = 1, m = 3,
                  condicionInicial = None,
                  a = 1.0,
                  b = 3.0,
                  c = 1.0,
                  d = 5.0,
                  s = 4.0,
                  r = 0.006,
                  x_r = -1.56,
                  I_bias = 2.9,
                  method = "RK45",
                  rtol = 1e-3,
                  atol = 1e-6):
    # Simulation conditions
    ##########################
    tiempo = np.linspace(0, t_final, int(t_final * ppt)) #ppt valores por segundo
    t_span = (0, t_final)
    if not isinstance(condicionInicial, np.ndarray):
        condicionInicial = np.random.rand(3*N)
    
#     if not isinstance(beta, np.ndarray):
#         if random_beta:
#             beta = (2 * np.random.rand(N) - 1) * std_beta
#         else:
#             beta = np.ones(N)
    
    if not isinstance(laplacianMatrix, np.ndarray):
        graph = nx.barabasi_albert_graph(N,m)
        laplacianMatrix = nx.laplacian_matrix(graph).todense()
    
    # Simulation
    ###########################
    integrator = scint.solve_ivp(HR_model, t_span, condicionInicial, args = (laplacianMatrix, g, a, b, c, d, s, r, x_r, I_bias), method = method, dense_output=True, rtol = rtol, atol = atol)
    solutions = integrator.sol(tiempo)
    # Each 3 rows is a neuron along time, each column is a timestamp for all neurons
    x = solutions[0::3]
    y = solutions[1::3]
    z = solutions[2::3]
    
    return x,y,z

def extreme_finder(x, type = "min"):
    peak_times = np.array(x)
    try:
        N, timestamps = np.shape(x)
    except:
        N = 1
        timestamps = len(x)
    for neuron in range(N):
        for t in range(timestamps):
            match type:
                case "min":
                    try:
                        # Not a minimum
                        if x[neuron, t] > x[neuron, t-1] or x[neuron, t] > x[neuron, t+1]:
                            peak_times[neuron, t] = 0
                    except:
                        # First and last points are not minima
                        peak_times[neuron, t] = 0
                case "max":
                    try:
                        # Not a maximum
                        if x[neuron, t] < x[neuron, t-1] or x[neuron, t] < x[neuron, t+1]:
                            peak_times[neuron, t] = 0
                    except:
                        # First and last points are not maxima
                        peak_times[neuron, t] = 0
                case default:
                    raise Exception("Enter a valid type (\'max\' or \'min\')")
    return np.where(peak_times != 0, 1, 0)

def kuramoto_parameter(solutions, ignore_first = 0.8):
    N, timestamps = np.shape(solutions)
    last_section_size = int(timestamps * ignore_first)
    return np.mean(np.sqrt(np.sum(np.cos(solutions[:, last_section_size:]), axis = 0)**2 + np.sum(np.sin(solutions[:, last_section_size:]), axis = 0)**2)/N)

def interpolation_with_poincare_section(x, valleys_df = None, verbose = False):
    poincare_phases = np.array(x)
    N, timestamps = np.shape(x)
    if not isinstance(valleys_df, pd.DataFrame):
        valleys_df = get_valleys_df(x)
    for neuron in range(N):
        if verbose and neuron % 25 == 0:
            print(f"Interpolating neuron {neuron}...")
        # k es el número de picos que llevamos
        k = 0
        number_of_peaks = valleys_df[(valleys_df["Neuron"] == neuron) & (valleys_df["Type"] == "Interburst")]["Neuron"].count()
        indices_of_peaks = valleys_df[(valleys_df["Neuron"] == neuron) & (valleys_df["Type"] == "Interburst")]["Time index"].to_numpy()
        for t in range(timestamps):
            #si el número de picos que llevamos es igual al número de picos que hay, salimos del bucle
            if k + 1 > number_of_peaks:
                break
            #si estamos entre picos o hemos alcanzado el pico al que nos acercábamos, calculamos la interpolación
            if k > 0 and t > indices_of_peaks[k-1]:
                poincare_phases[neuron, t] = 2*np.pi*(k-1) + 2*np.pi * (t - indices_of_peaks[k-1])/(indices_of_peaks[k] - indices_of_peaks[k-1])
            #si estamos en un pico, sumamos 1 al contador de picos que llevamos
            if t == indices_of_peaks[k]:        
                k+=1
    return poincare_phases

def synchronization_measure(solutions, method = "kura", interpolate = False, peaks = None, ignore_first = 0.8):
    # Synchronization of phases
    ###########################
    if method == "kura":
        if interpolate:
            poincare_phases = interpolation_with_poincare_section(solutions)
            kura = kuramoto_parameter(poincare_phases, ignore_first = ignore_first)
        else:
            kura = kuramoto_parameter(solutions, ignore_first = ignore_first)
        return kura
    if method == "std":
        N, timestamps = np.shape(solutions)
        last_section_size = int(timestamps * ignore_first)
        return np.mean(np.std(solutions[:, last_section_size:], axis = 0))

def cross_correlation_synchronization(time_series_data):
    N, T = time_series_data.shape
    total_corr = 0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            corr = np.corrcoef(time_series_data[i], time_series_data[j])[0, 1]
            total_corr += corr
            count += 1
    mean_corr = total_corr / count if count > 0 else 0
    return mean_corr

def get_shortest_time_periods(df : np.ndarray):
    # Get time difference with previous and next valleys
    df["Time diff with previous"] = df["Time"].diff()
    df["Time diff with next"] = df["Time"].diff(periods = -1)

    # We are including the time difference between the last valley of a neuron and
    # the first valley of the next one. We set it to 0.
    df.loc[df["Time diff with previous"] < 0, "Time diff with previous"] = 0
    df.loc[df["Time diff with next"] > 0, "Time diff with next"] = 0

    # Fill NaN
    df[["Time diff with previous", "Time diff with next"]] = df[["Time diff with previous", "Time diff with next"]].fillna(0)

    # Get the absolute value to compare both differences
    df["Time diff with next"] = df["Time diff with next"].apply(lambda x: abs(x))

    # Keep only the smallest difference.
    df["Min time diff"] = np.minimum(df["Time diff with previous"], df["Time diff with next"])
    df.drop(["Time diff with previous", "Time diff with next"], axis = 1, inplace = True)

    return

def separate_valleys_clustering(valleys_df, method = "gaussian_mixture", single_clustering = True, n_dim = 1, k = 2):
    valleys_df['Type'] = None
    if single_clustering:
        if n_dim == 1:
            data = valleys_df["Min time diff"].values.reshape(-1, 1)
        else:
            data = valleys_df[["x", "Min time diff"]].values.reshape(-1, 2)
        if method == "gaussian_mixture":
            valleys_df["Label"] = GaussianMixture(n_components=k, init_params="k-means++", n_init = 12).fit_predict(data)
        if method == "kmeans":
            kmeans = KMeans(n_clusters=k, n_init = 30).fit(data)
            valleys_df["Label"] = kmeans.labels_
        if method == "birch":
            valleys_df["Label"] = Birch(n_clusters=k, threshold=0.1).fit_predict(data)
        inter_label, intra_label = valleys_df[["x", "Label"]].groupby("Label").mean("x").sort_values(by = "x", ascending = True).index
        valleys_df['Type'] = valleys_df['Type'].astype(str)
        valleys_df.loc[valleys_df["Label"] == inter_label, "Type"] = "Interburst"
        valleys_df.loc[valleys_df["Label"] == intra_label, "Type"] = "Intraburst"
    else:
        for neuron in valleys_df["Neuron"].unique():
            if n_dim == 1:
                data = valleys_df.loc[valleys_df["Neuron"] == neuron, "Min time diff"].values.reshape(-1, 1)
            else:
                data = valleys_df.loc[valleys_df["Neuron"] == neuron, ["x", "Min time diff"]].values.reshape(-1, 2)
            if method == "gaussian_mixture":
                valleys_df.loc[valleys_df["Neuron"] == neuron, "Label"] = GaussianMixture(n_components=k, init_params="k-means++", n_init = 20).fit_predict(data)
            if method == "kmeans":
                kmeans = KMeans(n_clusters=k, n_init = "auto").fit(data)
                valleys_df.loc[valleys_df["Neuron"] == neuron, "Label"] = kmeans.labels_
            if method == "birch":
                valleys_df.loc[valleys_df["Neuron"] == neuron, "Label"] = Birch(n_clusters=k, threshold=0.1).fit_predict(data)
            inter_label, intra_label = valleys_df.loc[valleys_df["Neuron"] == neuron, ["x", "Label"]].groupby("Label").mean("x").sort_values(by = "x", ascending = True).index
            valleys_df.loc[valleys_df["Neuron"] == neuron, 'Type'] = valleys_df.loc[valleys_df["Neuron"] == neuron, 'Type'].astype(str)
            valleys_df.loc[(valleys_df["Neuron"] == neuron) & (valleys_df["Label"] == inter_label), "Type"] = "Interburst"
            valleys_df.loc[(valleys_df["Neuron"] == neuron) & (valleys_df["Label"] == intra_label), "Type"] = "Intraburst"
    
    valleys_df.drop("Label", axis = 1, inplace = True)
    return

# Cuenta los mínimos que hay entre este mínimo interburst y el anterior.
def get_number_of_valleys_per_burst(valleys_df, clean = False):
    interburst_mask = (valleys_df["Type"] == "Interburst") | (((valleys_df["Min time diff"] == 0) | (valleys_df["Min time diff"].isna())) & (valleys_df["Time"] < 1000))
    valleys_df["Group"] = interburst_mask.cumsum()
    valleys_df["Number of valleys"] = np.nan
    valleys_df.loc[valleys_df["Type"] == "Interburst", "Number of valleys"] = 0
    intraburst_counts = valleys_df.groupby("Group").size() - 1
    valleys_df.loc[interburst_mask, 'Number of valleys'] = valleys_df.loc[interburst_mask, 'Group'].map(intraburst_counts)
    # valleys_df.loc[(valleys_df["Type"] == "Interburst") & (valleys_df["Interburst time"].isna()), "Number of valleys"] = 0
    valleys_df.loc[valleys_df["Type"] == "Intraburst", "Number of valleys"] = np.nan
    valleys_df.drop("Group", axis = 1, inplace = True)
    # Get rid of the bursts with 0 peaks
    if clean:
        valleys_df.drop(valleys_df[(valleys_df["Number of valleys"] == 0)].index, axis = 0, inplace = True)
    return

def get_number_of_peaks_per_burst(peaks_df):
    interburst_mask = (peaks_df["Type"] == "Interburst") | (((peaks_df["Min time diff"] == 0) | (peaks_df["Min time diff"].isna())) & (peaks_df["Time"] < 1000))
    peaks_df["Group"] = interburst_mask.cumsum()
    peaks_df["Number of spikes"] = np.nan
    peaks_df.loc[peaks_df["Type"] == "Interburst", "Number of spikes"] = 0
    peak_counts = peaks_df[peaks_df["Type"] == "Spike"].groupby("Group").size()
    peak_counts = peaks_df.groupby("Group").size() - 1
    peaks_df.loc[interburst_mask, 'Number of spikes'] = peaks_df.loc[interburst_mask, 'Group'].map(peak_counts)
    peaks_df.loc[(peaks_df["Type"] == "Interburst") & (peaks_df["Interburst time"].isna()), "Number of spikes"] = np.nan
    peaks_df.loc[peaks_df["Type"] == "Spike", "Number of spikes"] = np.nan
    peaks_df.drop("Group", axis = 1, inplace = True)
    peaks_df.drop(peaks_df[peaks_df["Number of spikes"] == 0].index, axis = 0, inplace = True)
    return


def get_valleys_df(x, tiempo, method = "gaussian_mixture", manual_threshold = -1.15, single_clustering = True, n_dim = 1, only_manual = False):
    valleys = extreme_finder(x)
    neuron_indices, time_indices = np.nonzero(valleys == 1)
    valleys_df = pd.DataFrame({
        "Neuron" : pd.Series(neuron_indices),
        "Time index" : pd.Series(time_indices),
        "Time" : pd.Series(tiempo[time_indices]),
        "x" : pd.Series(x[neuron_indices, time_indices])
    })
    get_shortest_time_periods(valleys_df)

    #Initial identification of minima
    valleys_df["Type"] = valleys_df["x"].apply(lambda x: "Interburst" if x <= manual_threshold else "Intraburst")

    # Calculate burst duration
    valleys_df["Interburst time"] = valleys_df[valleys_df["Type"] == "Interburst"]["Time"].diff()

    # Some valleys have negative time differences because they belong to different neurons. Set them to NaN
    valleys_df.loc[(valleys_df["Type"] == "Interburst") & (valleys_df["Interburst time"] < 0), "Interburst time"] = np.nan

    # Number of valleys per burst. It also removes the bursts with no peaks
    get_number_of_valleys_per_burst(valleys_df, clean=True)
    get_shortest_time_periods(valleys_df)

    if only_manual:
        return valleys_df

    # Repeat the process, now with an unsupervised clustering method
    # Cluster with unsupervised method instead of a threshold
    separate_valleys_clustering(valleys_df, method = method, single_clustering=single_clustering, n_dim = n_dim)

    # Repeat the rest of steps
    valleys_df["Interburst time"] = valleys_df[valleys_df["Type"] == "Interburst"]["Time"].diff()
    valleys_df.loc[(valleys_df["Type"] == "Interburst") & (valleys_df["Interburst time"] < 0), "Interburst time"] = np.nan
    get_number_of_valleys_per_burst(valleys_df)

    return valleys_df
 
def get_peaks_df(x : np.ndarray, tiempo : np.ndarray, valleys_df : np.ndarray) -> np.ndarray:
    peaks = extreme_finder(x, type = "max")
    neuron_indices, time_indices = np.nonzero(peaks == 1)
    peaks_df = pd.DataFrame({
        "Neuron" : pd.Series(neuron_indices),
        "Time index" : pd.Series(time_indices),
        "Time" : pd.Series(tiempo[time_indices]),
        "x" : pd.Series(x[neuron_indices, time_indices])
    })
    peaks_df.loc[:, "Type"] = "Spike"

    # Localize the burst intervals
    peaks_df = pd.concat([peaks_df, valleys_df[valleys_df["Type"] == "Interburst"]], ignore_index=True).sort_values(by = ["Neuron", "Time index"]).reset_index().loc[:, ["Neuron", "Time index", "Time", "x", "Type", "Interburst time"]]
    min_possible_peak_value = valleys_df.loc[valleys_df["Type"] == "Intraburst", "x"].min()
    peaks_df.drop(peaks_df.loc[(peaks_df["Type"] == "Spike") & (peaks_df["x"] <= min_possible_peak_value)].index, axis = 0, inplace = True)

    get_shortest_time_periods(peaks_df)
    get_number_of_peaks_per_burst(peaks_df)

    return peaks_df

# Reads a graph in JSON format and returns it as a nx graph
def retrieve_network(N : int, m : int, i : int = 0, path : str = 'networks') -> nx.graph:
    with open(f"{path}/network_N_{N}_m_{m}_{i}.json", "r") as fh:
        serialized_graph = fh.read()
    return nx.node_link_graph(json.loads(serialized_graph))

if __name__ == '__main__':
    x = np.random.rand(2, 100)
    tiempo = np.linspace(0, 100, 100)
    valleys_df = get_valleys_df(x, tiempo)
    peaks_df = get_peaks_df(x, tiempo, valleys_df)

    print(peaks_df)


