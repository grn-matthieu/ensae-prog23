# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import graph_from_file
import unittest   # The test framework

class Test_Kruskal(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("input/network.1.in")
        g = g.kruskal()
        self.assertEqual(g.nb_edges, 19)

    def test_network4(self):
        g = graph_from_file("input/network.4.in")
        g = g.kruskal()
        self.assertEqual(g.nb_edges, 99999)

if __name__ == '__main__':
    unittest.main()
