import graphviz
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
            #Little change test
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

    def get_list_of_paths_with_power(self, node, dest, power=-1, seen=[], liste=[]):
        if node not in seen:
            #If the edge is not in the set, we add it and depth search from it.
            seen.append(node)
            if node == dest:
                liste.append(seen)
            list_of_neighbours = self.list_of_neighbours(node)
            for index, neighbor in enumerate(list_of_neighbours):
                if self.graph[node][index][1] > power and power!=-1:
                    continue
                self.get_list_of_paths_with_power(neighbor,dest,seen=seen.copy(),liste=liste)
        return liste

    def get_path_with_power(self,origin,destination,power=-1):
        list_of_paths = self.get_list_of_paths_with_power(origin,destination,power)
        if list_of_paths == []:
            return None
        if len(list_of_paths) == 1:
            return list_of_paths[0]
        min_index = self.minimum_distance(origin,destination,list_of_paths)
        return list_of_paths[min_index]

    def minimum_distance(self,origin, destination, possible_paths=[]):
        list_of_neighbours = [self.list_of_neighbours(node) for node in self.nodes if self.graph[node]!=[]]
        current_distance = 0
        distance = 0
        for index, path in enumerate(possible_paths):
            for node in path:
                if node == origin:
                    previous_node = node
                    continue
                index_previous_node = list_of_neighbours[node-1].index(previous_node)
                current_distance += self.graph[node][index_previous_node][2]
                previous_node = node
            if current_distance<distance or distance==0:
                distance = current_distance
                min_index = index            
        return min_index
    
    def min_power(self,origin,destination):
        list_of_paths = self.get_list_of_paths_with_power(origin,destination)
        list_of_neighbours = [self.list_of_neighbours(node) for node in self.nodes if self.graph[node]!=[]]
        min_power = 0
        for index, path in enumerate(list_of_paths):
            for node in path:
                if node == origin:
                    previous_node = node
                    power_of_path = 0
                    continue
                index_previous_node = list_of_neighbours[node-1].index(previous_node)
                power_of_path = max(self.graph[node][index_previous_node][1],power_of_path)
                previous_node = node
            if min_power<power_of_path or min_power == 0:
                min_power = power_of_path
                min_power_path = path
        return (min_power_path,min_power)

    def list_of_neighbours(self, node):
        """
        Returns a list of nodes connected to the input node.

        Parameters:
        -----------
        node : nodeType(integer)
            The node for which we consider neighbours.
        """
        return list(zip(*self.graph[node]))[0]

    def depth_search(self, node, seen=None, power=-1, dest=0):
        """
        Deep searches a graph, and returns a list of connected nodes.

        Parameters:
        -----------
        node : NodeType(integer)
            The starting node for the algorithm to run.
        seen : set
            The set which contains all of the connected nodes.
        
        The complexity of this algorithm is O(|V| + |E|), where |V| is the number of nodes, and |E| is the number of edges.
        """
        #We implement a recursive function to depth search our graph.
        if seen == None:
            #In the case of first implementation, we create the set.
            seen = set()
        if node not in seen:
            #If the edge is not in the set, we add it and depth search from it.
            seen.add(node)
            list_of_neighbours = self.list_of_neighbours(node)
            for index, neighbor in enumerate(list_of_neighbours):
                if self.graph[node][index][2] > power and power!=-1:
                    #If we use the algorithm with a given power, we check if two nodes can be linked with this power
                    #If not, we do not consider the neighbor.
                    continue
                self.depth_search(neighbor,seen,power=power)
        return seen

    def connected_components(self):
        """
        Builds a list of connected nodes, grouped in lists, and returns it.
        This function relies on the depth-first search alogrithm, implemented in depth_search().

        """
        connected_list = []
        nodes_to_see = self.nodes.copy()
        for node in nodes_to_see:
            #We check all existing nodes in our graph.
            new_graph = sorted(self.depth_search(node))
            #We sort the list of connected components in order to compare lists of connected components.
            if new_graph not in connected_list:
                connected_list.append(new_graph)
                for node in new_graph:
                    #We make sure not to run the DFS on a node that has already been detected
                    nodes_to_see.remove(node)
        return connected_list

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
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

def graph_into_pdf(filename):
    file = open(filename, 'r')
    graph = graphviz.Digraph('Le graphe')
    dist=1
    #First line is read in order to properly intialize our graph
    line_1 = file.readline().split(' ')
    #Then, all lines are read to create a new edge for each line
    for line in file:
        list_line = line.split(' ')
        if list_line == []:
            continue
        if len(list_line) == 4:
            #In the case where a distance is included
            dist = int(list_line[3])
        graph.node(list_line[0])
        graph.node(list_line[1])
        graph.edge(list_line[0],list_line[1],arrowhead='none')
    file.close()
    graph.render()