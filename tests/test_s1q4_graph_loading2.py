# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import Graph, graph_from_file

import unittest   # The test framework

class Graph_Loading(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        self.assertEqual(g.graph[1][0][2], 1)
    
    def test_network1(self):
        h = graph_from_file("input/network.1.in")
        self.assertEqual(h.graph[1][0][2], 6312)
        self.assertEqual(h.graph[16][4][2], 9032)

if __name__ == '__main__':
    unittest.main()
