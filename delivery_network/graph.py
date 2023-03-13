import graphviz
class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.list_of_neighbours = []
        self.list_of_edges = []
        self.max_power = 0
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
    
    def add_edge(self, node1, node2, power_min, dist=1, index=-1):
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

        #We build a list of all edges
        self.list_of_edges.append((node1,node2,power_min))

    def minimum_distance(self,origin, destination, possible_paths=[]):
        '''
        This function returns the minimum distance to link two nodes.
        In order to do so, we examine all possible paths, retrieve the total distance for each, and choose the minimum

        Parameters:
        -----------
            -origin : Nodetype(integer)
                The starting node for the path
            -destination : NodeType(integer)
                The destination node
            -possible_paths : list
                the list of all possible paths to link the two nodes together

        Output:
        -------
            -min_index : The index for the smallest distance to link the nodes together
        '''
        list_of_neighbours = self.list_of_neighbours
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
        '''
        This function searches the minimum power to link two nodes together.
        It relies on binary search : we list all powers in our graph, and we look if the nodes can be linked with a given power.

        Parameters:
        -----------
            -origin : NodeType(integer)
                The starting node.
            -destination : NOdeType(integer)
                The end node.

        Output :
        --------
            -(power,path):
                A tuple containing the minimum power, and the associated path.     
        '''
        path=[]
        start = 0
        end = self.max_power
        if destination not in self.depth_search(origin):
            return None
        while start != end:
            mid = (start+end)//2
            if destination not in self.depth_search(origin, power=mid):
                start = mid
            else:
                end = mid
            if end-start == 1:
                start=end
        #We now have identified the minimum power to link the two nodes, stored in power
        #We have to get the path
        power = end
        path = self.get_list_of_paths_with_power(origin,destination,power=power)[0]
        return (power,path)
        
    def get_list_of_paths_with_power(self, node, dest, power=-1, seen=[], liste=[]):
        '''
        This function searches for all of the possible paths between two nodes, given a certain power(optional).
        It relies on a DFS, which is modified to remmeber the path that has been taken.
        This function is recursive.

        Parameters:
        ----------
            -node : NodeType(integer)
                The starting node.
            -dest : NodeType(integer)
                The end node.
            -power : integer
                The optional power which can restrain the available edges.
            -seen : list
                The list of the nodes that have been seen by the algorithm.
            -liste : liste
                The list containing the possible paths.
    
        Output:
        -------
            -liste : liste
                The list containing all of the possible paths.
        '''
        if node not in seen:
            seen.append(node)
            if node == dest:
                liste.append(seen)
                #A MODIFIER : ON BLOQUE A UN SEUL TRAJET
                return liste
            for index, neighbor in enumerate(self.list_of_neighbours[node-1]):
                if self.graph[node][index][1] > power and power!=-1:
                    continue
                self.get_list_of_paths_with_power(neighbor,dest,seen=seen.copy(),liste=liste)
        return liste

    def get_path_with_power(self,origin,destination,power=-1):
        '''
        This function returns a path compatible with a given power between two nodes.

        Parameters:
        -----------
            -origin : nodeType(integer)
                The starting node for our algorithm
            -destination : nodeType(integer)
                The destination node.
            -power : integer
                The optional power which can restrain the edges taken by the algorithm.

        Output:
        -------
            -list_of_paths[integer] : a path from the list of all possible paths.
        
        '''
        list_of_paths = self.get_list_of_paths_with_power(origin,destination,power)
        if list_of_paths == []:
            return None
        if len(list_of_paths) == 1:
            return list_of_paths[0]
        min_index = self.minimum_distance(origin,destination,list_of_paths)
        return list_of_paths[min_index]
    
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
            list_of_neighbours = self.list_of_neighbours
            for index, neighbor in enumerate(list_of_neighbours[node-1]):
                if self.graph[node][index][1] > power and power!=-1:
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

    def lowest_common_ancestor(self, node1, node2):
        """
        Finds the lowest common ancestor of two nodes in a minimum spanning tree.

        Parameters:
        -----------
        node1: NodeType
            The starting node.
        node2: NodeType
            The end node.

        Returns:
        --------
        lowest_common_ancestor: NodeType
            The lowest common ancestor of the two nodes.
        """
        print(dict_of_nodes[1])


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
    start = time.perf_counter()
    file = open(filename, 'r')
    dist=1
    #First line is read in order to properly intialize our graph
    line_1 = file.readline().split(' ')
    total_nodes = int(line_1[0])
    nb_edges = int(line_1[1].strip('\n'))
    new_graph = Graph([node for node in range(1,total_nodes+1)])
    new_graph.nb_edges = nb_edges
    #new_graph.list_of_edges = [None]*nb_edges
    #Then, all lines are read to create a new edge for each line
    for line in file:
        list_line = line.split(' ')
        start_node = int(list_line[0])
        end_node = int(list_line[1])
        power = int(list_line[2])
        if list_line == []:
            continue
        if len(list_line) == 4:
            #In the case where a distance is included
            dist = int(list_line[3])
        new_graph.max_power = max(new_graph.max_power, power)
        new_graph.add_edge(start_node, end_node, power, dist)
    new_graph.list_of_neighbours = [list(zip(*new_graph.graph[node]))[0] for node in new_graph.nodes if new_graph.graph[node]!=[]]
    stop = time.perf_counter()
    print(f'Graphe tracé en {stop-start} secondes')
    file.close()
    return new_graph


class Union_Find():
    '''
    This class is used later to implement the Kruskal's algorithm.
    It allows to create a Union-Find structure.
    The path compression upgrade of the structure allows us to reduce its complexity.
    '''
    def __init__(self):
        self.rank = -1
        self.parent = -1

    def make_set(self):
        self.parent = self
        self.rank = 0
    
    def find(self):
        while self != self.parent:
            self = self.parent
        return self
    
    def union(self, y):
        root_x = self.find()
        root_y = y.find()    
        if root_x == root_y :
            return 
        if root_x.rank > root_y.rank:
            root_y.parent = root_x
        else:
            root_x.parent = root_y
            if root_x.rank == root_y.rank:
                root_y.rank = root_y.rank + 1


def kruskal(input_graph):
    '''
    This function is an implementation of the Kruskal's algorithm, described in Algorithms, Dasgupta et al.

    Parameters:
    -----------
    input_graph : graph object
        The graph which will be spanned by the algorithm.

    Output:
    -------
    output_graph : graph object
        The minimum spanning tree extracted from input_graph.   
    '''
    #We need to build a list of all edges, sorted by power, and in order that they are not duplicated
    #The list is partially build thanks to add_edge function, but it needs to be sorted
    sorted_edges = input_graph.list_of_edges
    sorted_edges.sort(key=lambda a : a[2])
    #The list of edges sorted by power_min is now stored in sorted_edges
    global dict_of_nodes
    dict_of_nodes = {}
    for node in input_graph.nodes:#We create a list of nodes in our union find structure
        dict_of_nodes[node] = Union_Find()
        dict_of_nodes[node].make_set()
    X = set()
    for edge in sorted_edges:
        #We start by identifying our linked nodes
        node_1 = edge[0]
        node_2 = edge[1]
        power_min = edge[2]
        if dict_of_nodes[node_1].find() != dict_of_nodes[node_2].find():#If the two nodes do not share the same parent, we add the edge to the set
            X.add((node_1,node_2,power_min))
            dict_of_nodes[node_1].union(dict_of_nodes[node_2])#We link the nodes together in order to update nodes' parents
    start = time.perf_counter()
    #X now contains the list of all edges of the spanned graph. We have to create it.
    output_graph = Graph(input_graph.nodes)
    output_graph.nb_edges = input_graph.nb_nodes - 1
    for edge in X:
        origin, destination, power = edge[0], edge[1], edge[2]
        #We look at all edges to properly intialize our graph object.
        output_graph.add_edge(origin,destination,power)
        output_graph.list_of_edges.append(edge)
        output_graph.max_power = max(output_graph.max_power, power)
    output_graph.list_of_neighbours = [list(zip(*output_graph.graph[node]))[0] for node in output_graph.nodes if output_graph.graph[node]!=[]]
    #All of the graph parameters are set :)
    stop = time.perf_counter()
    print(f'Graphe par Kruskal tracé en {stop-start} secondes')
    return output_graph

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
    start = time.perf_counter()
    file = open(filename, 'r')
    dist=1
    #First line is read in order to properly intialize our graph
    line_1 = file.readline().split(' ')
    total_nodes = int(line_1[0])
    nb_edges = int(line_1[1].strip('\n'))
    new_graph = Graph([node for node in range(1,total_nodes+1)])
    new_graph.nb_edges = nb_edges
    new_graph.list_of_edges = [None]*nb_edges
    #Then, all lines are read to create a new edge for each line
    for line in file:
        list_line = line.split(' ')
        start_node = int(list_line[0])
        end_node = int(list_line[1])
        power = int(list_line[2])
        if list_line == []:
            continue
        if len(list_line) == 4:
            #In the case where a distance is included
            dist = int(list_line[3])
        new_graph.max_power = max(new_graph.max_power, power)
        new_graph.add_edge(start_node, end_node, power, dist)
    new_graph.list_of_neighbours = [list(zip(*new_graph.graph[node]))[0] for node in new_graph.nodes if new_graph.graph[node]!=[]]
    stop = time.perf_counter()
    print(stop-start)
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