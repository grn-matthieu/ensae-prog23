from graph import Graph, graph_from_file


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
a = g.connected_components()
print(a)