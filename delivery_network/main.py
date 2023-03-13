from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, kruskal, find_path_with_kruskal
import os
import time
import random

data_path = "input/"
file_name = "network.1.in"

def estimate_time(graph):
    list_of_times = []
    N=20
    list_of_paths = []
    for index in range(N):
        list_of_paths.append((random.randint(1,g.nb_nodes),random.randint(1,g.nb_nodes)))
    print(list_of_paths)

    for origin,destination in list_of_paths:
        current_time_start = time.perf_counter()
        graph.min_power(origin,destination)
        current_time_stop = time.perf_counter()
        list_of_times.append(current_time_stop-current_time_start)
    print(f'Moyenne de temps de traitement : {(1/N)*sum(list_of_times)}')


h = graph_from_file(data_path + file_name)
g = kruskal(h)
print(find_path_with_kruskal(g, 1, 20))