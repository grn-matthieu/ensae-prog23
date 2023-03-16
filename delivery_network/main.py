from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, route_min_power, build_m_matrix
import os
import time
import random
import numpy as np

data_path = "input/"
file_name = "network.1.in"

def estimate_time():
    '''
    This function estimates the time to determine min_power for N paths.
    To do so, it calculate nb_paths times min_power, and means it to determine time recquired for 1 path.

    The results show that for big graphs, we need approx 2 min to calculate 500 000 paths.
    In this version, we rely on a kruskal graph and the find_path_with_kruskal, but we might also test with min_power
    to show the time difference.
    '''
    start_tot = time.perf_counter()
    graph_1 = graph_from_file(data_path+file_name)
    graph_2 = graph_1.kruskal()
    graph_2.build_parents()
    list_of_times = []
    N=10000
    nb_paths = 500000
    list_of_paths = []
    for index in range(N):
        list_of_paths.append((random.randint(1,graph_2.nb_nodes),random.randint(1,graph_2.nb_nodes)))
    for origin,destination in list_of_paths:
        current_time_start = time.perf_counter()
        graph_2.find_path_with_kruskal(origin,destination)
        current_time_stop = time.perf_counter()
        list_of_times.append(current_time_stop-current_time_start)
    print(f'Moyenne de temps de traitement du trajet: {(1/N)*sum(list_of_times)}')
    tps = nb_paths*(1/N)*sum(list_of_times)
    print(f'Pour {nb_paths} trajet : {tps} secondes')
    stop_tot = time.perf_counter()
    print(f'Temps total : {stop_tot+tps-start_tot}')

print(build_m_matrix(1))