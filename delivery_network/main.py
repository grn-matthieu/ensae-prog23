from graph import Graph, graph_from_file, graph_into_pdf
import os


data_path = "input/"
file_name = "network.1.in"

g = graph_from_file(data_path + file_name)
print(g.min_power(1,2))
print(g.depth_search(1,power=57))
print(g.depth_search(1,power=58))