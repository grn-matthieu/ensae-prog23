from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, kruskal, find_path_with_kruskal, determine_parents, route_min_power
import os
import time
import random

data_path = "input/"
file_name = "network.4.in"

def estimate_time():
    start_tot = time.perf_counter()
    graph_1 = graph_from_file(data_path+file_name)
    graph_2 = kruskal(graph_1)
    determine_parents(graph_2)
    list_of_times = []
    N=1000
    nb_trajets = 500000
    list_of_paths = []
    for index in range(N):
        list_of_paths.append((random.randint(1,graph_2.nb_nodes),random.randint(1,graph_2.nb_nodes)))
    for origin,destination in list_of_paths:
        current_time_start = time.perf_counter()
        find_path_with_kruskal(graph_2,origin,destination)
        current_time_stop = time.perf_counter()
        list_of_times.append(current_time_stop-current_time_start)
    print(f'Moyenne de temps de traitement du trajet: {(1/N)*sum(list_of_times)}')
    tps = nb_trajets*(1/N)*sum(list_of_times)
    print(f'Pour {nb_trajets} trajet : {tps} secondes')
    stop_tot = time.perf_counter()
    print(f'Temps total : {stop_tot+tps-start_tot}')

estimate_time()