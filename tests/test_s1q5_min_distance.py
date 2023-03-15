# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import Graph, graph_from_file

import unittest   # The test framework

class Minimum_Distance(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        self.assertEqual(g.minimum_distance(1,4,11)[1], 3)
        self.assertEqual(g.get_path_with_power(1, 4, 10), 0)

    def test_network01(self):
        h = graph_from_file("input/network.01.in")
        self.assertEqual(h.minimum_distance(1, 2, 11)[1], 1)
        self.assertEqual(h.get_path_with_power(1, 7, 5), 0)

if __name__ == '__main__':
    unittest.main()
