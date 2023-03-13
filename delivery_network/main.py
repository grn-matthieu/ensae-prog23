from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, kruskal, find_path_with_kruskal, update_parents
import os
import time
import random

data_path = "input/"
file_name = "network.1.in"

def estimate_time():
    graph_1 = graph_from_file(data_path+file_name)
    graph_2 = kruskal(graph_1)
    list_of_times = []
    N=20
    list_of_paths = []
    for index in range(N):
        list_of_paths.append((random.randint(1,graph_2.nb_nodes),random.randint(1,graph_2.nb_nodes)))
    print(list_of_paths)

    for origin,destination in list_of_paths:
        current_time_start = time.perf_counter()
        find_path_with_kruskal(graph_2,origin,destination)
        current_time_stop = time.perf_counter()
        list_of_times.append(current_time_stop-current_time_start)
    print(f'Moyenne de temps de traitement : {(1/N)*sum(list_of_times)}')

g = graph_from_file(data_path + file_name)
h = kruskal(g)
update_parents(h)
start = time.perf_counter()
print(find_path_with_kruskal(h, 11, 15))
stop = time.perf_counter()
print(stop-start)
start = time.perf_counter()
print(h.min_power(11,15))
stop = time.perf_counter()
print(stop-start)