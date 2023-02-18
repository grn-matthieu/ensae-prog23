class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        """
        Example graph format for two nodes(1,2) connected:
        self.nodes = [1,2]
        self.graph = {1:[(2, power, dist)], 2:[(1, power, dist)]}
        self.nb_nodes = 2
        self.nb_edges = 1
        """

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        #We can use the append function as graph_from_file automatically creates the graph with all of our nodes
        self.graph[node1].append((node2,power_min,dist))
        self.graph[node2].append((node1,power_min,dist))


    def get_path_with_power(self, src, dest, power):
        raise NotImplementedError

    def depth_search(self, node, seen=None):
        """
        Deep searches a graph, and returns a list of connected nodes.

        Parameters:
        -----------
        node : NodeType(integer)
            The starting node for the algorithm to run.
        seen : set
            The set which contains all of the connected nodes.
        """
        #We implement a recursive function to depth search our graph.
        if seen == None:
            #In the case of first implementation, we create the set.
            seen = set()
        if node not in seen:
            #If the edge is not in the set, we add it and depth search from it.
            seen.add(node)
            list_of_neighbours = list(zip(*self.graph[node]))[0]
            for neighbor in list_of_neighbours:
                self.depth_search(neighbor,seen)
        return seen

    def connected_components(self):
        """
        Builds a list of connected nodes, grouped in lists, and returns it.
        This function relies on the depth search alogrithm, implemented in depth_search().

        """
        connected_list = []
        for node in self.nodes:
            #We check all existing nodes in our graph.
            new_graph = sorted(self.depth_search(node))
            #We sort the list of connected components in order to compare lists of connected components.
            if new_graph not in connected_list:
                connected_list.append(new_graph)
        return connected_list



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        raise NotImplementedError


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    file = open(filename, 'r')
    dist=1
    #First line is read in order to properly intialize our graph
    line_1 = file.readline().split(' ')
    new_graph = Graph([node for node in range(1,int(line_1[0])+1)])
    new_graph.nb_edges = int(line_1[1].strip('\n'))
    #Then, all lines are read to create a new edge for each line
    for line in file:
        list_line = line.split(' ')
        if list_line == []:
            continue
        if len(list_line) == 4:
            #In the case where a distance is included
            dist = int(list_line[3])
        new_graph.add_edge(int(list_line[0]), int(list_line[1]), int(list_line[2]),dist)
    file.close()
    return new_graph