from graph import Graph, Union_Find, graph_from_file, graph_into_pdf, kruskal
import os
import graphviz
import time
import random

data_path = "input/"
file_name = "network.02.in"

g = graph_from_file(data_path + file_name)
h = kruskal(g)
print(g)
print(h)
'''list_of_times = []
def estimate_time():
    N=20
    list_of_paths = []
    for index in range(N):
        list_of_paths.append((random.randint(1,g.nb_nodes),random.randint(1,g.nb_nodes)))
    print(list_of_paths)

    for origin,destination in list_of_paths:
        current_time_start = time.perf_counter()
        g.min_power(origin,destination)
        current_time_stop = time.perf_counter()
        print(current_time_start)
        print(current_time_stop)
        list_of_times.append(current_time_stop-current_time_start)
    print(f'Moyenne de temps de traitement : {(1/N)*sum(list_of_times)}')
estimate_time()'''
