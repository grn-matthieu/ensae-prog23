# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import graph_from_file
import unittest   # The test framework

class Test_MinimalPower(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("input/network.1.in")
        g = g.kruskal()
        g.build_parents()
        self.assertEqual(g.find_path_with_kruskal(1,2), (2, [1, 2]))
        self.assertEqual(g.find_path_with_kruskal(1,10),(37, [1, 14, 7, 16, 10]))

if __name__ == '__main__':
    unittest.main()
