from graph import Graph, graph_from_file, graph_into_pdf
from graphviz import *
import os


data_path = "input/"
file_name = "network.00.in"

g = graph_from_file(data_path + file_name)

graph_into_pdf(data_path+file_name)