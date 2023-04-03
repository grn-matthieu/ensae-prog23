import graphviz
import time
import sys
import numpy as np
sys.setrecursionlimit(10000)#In this programm, we use recursive DFS a lot
class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.list_of_neighbours = []
        self.list_of_edges = []
        self.max_power = 0
        self.parent_list = []
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

    def minimum_distance(self,origin, destination,power):
        '''
        This function returns the minimum distance to link two nodes.
        In order to do so, we examine all possible paths, retrieve the total distance for each, and choose the minimum

        Parameters:
        -----------
            -origin : Nodetype(integer)
                The starting node for the path
            -destination : NodeType(integer)
                The destination node
            -power : integer
                The max power which our path can take.

        Output:
        -------
            -min_index : The index for the smallest distance to link the nodes together
        '''
        possible_paths = self.get_list_of_paths_with_power(origin,destination,power)
        list_of_neighbours = self.list_of_neighbours
        current_distance = 0
        distance = 0
        #We have listed all possible paths.
        for index, path in enumerate(possible_paths):#We look at all paths to determine the minimum distance.
            for index_2, node in enumerate(path):
                if node == path[-1]:
                    continue
                neighbour_index = self.list_of_neighbours[node-1].index(path[index_2+1])
                current_distance += self.graph[node][neighbour_index][2]
                previous_node = node
            if current_distance<distance or distance==0:
                distance = current_distance
                min_index = index
        if distance == 0:
            return 0
        return (possible_paths[min_index], distance)

    def min_power(self,origin,destination):
        '''
        This function searches the minimum power to link two nodes together.
        It relies on binary search : we list all powers in our graph, and we look if the nodes can be linked with a given power.
        In the worst case scenario, its complexity is log_2(max_power) + 1.

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
        
    def get_list_of_paths_with_power(self, node, dest, power=-1, seen=[], liste=[], unique=False):
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
            -liste : List
                The list containing all of the possible paths.
        '''
        #This function is a "classic" DFS.
        if node not in seen:
            seen.append(node)
            if node == dest:
                liste.append(seen)
                if unique == True:#If we only want one path among all.
                    return seen
            for index, neighbor in enumerate(self.list_of_neighbours[node-1]):
                if self.graph[node][index][1] > power and power!=-1:
                    continue
                self.get_list_of_paths_with_power(neighbor,dest,seen=seen.copy(),liste=liste, unique=unique)
        return liste

    def get_path_with_power(self,origin,destination,power=-1):
        '''
        This function returns a path compatible with a given power between two nodes.
        This fuction relies on a DFS which is unique given the fact that we need a unique path.
        Therefore, its complexity is O(|V| + |E|), where |V| is the number of nodes, and |E| is the number of edges.

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
        liste = []
        seen = []
        list_of_paths = self.get_list_of_paths_with_power(origin,destination,power, seen = seen, liste = liste, unique=True)
        if list_of_paths == []:
            return None
        return list_of_paths[0]
    
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

    def kruskal(self):
        '''
        This function is an implementation of the Kruskal's algorithm, described in Algorithms, Dasgupta et al.
        Its complexity is O(E*V), given the fact that our union-find structure relies on sets.

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
        #The list is partially built thanks to add_edge function, but it needs to be sorted
        sorted_edges = self.list_of_edges
        sorted_edges.sort(key=lambda a : a[2])
        #The list of edges sorted by power_min is now stored in sorted_edges
        output_graph = Graph(self.nodes)
        dict_of_nodes = {}
        for node in self.nodes:#We create a list of nodes in our union find structure
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
        output_graph.nb_edges = self.nb_nodes - 1
        for edge in X:
            origin, destination, power = edge[0], edge[1], edge[2]
            #We look at all edges to properly intialize our graph object.
            output_graph.add_edge(origin,destination,power_min=power)
            output_graph.max_power = max(output_graph.max_power, power)
        output_graph.list_of_neighbours = [list(zip(*output_graph.graph[node]))[0] if self.graph[node]!=[] else () for node in self.nodes]
        #All of the graph parameters are set :)
        stop = time.perf_counter()
        print(f'Graphe par Kruskal tracé en {stop-start} secondes')
        return output_graph
    
    def build_parents(self):
        '''
        This function properly determines the parent-children system for a graph.
        It then relies on the determine function, which is a DFS.
        We must travel through the whole graph, and because the graph is a minimum spanning tree, E = V -1
        The complexity of this algorithm is O(V).

        Parameters:
        -----------
            -input_graph : GraphType
                The input graph which we process. It is necessary for this graph to have been processed by the Kruskal algorithm.
        
        Output:
        -------
            -self.parents : List
                The list which resumes the parent-children relationship between nodes.
        '''
        liste_parents = list(range(1,self.nb_nodes+1))#We define node n°1 to be the highest ancestor, but it could be any node.
        self.parent_list =  self.parents(1, liste_parents=liste_parents)#We build self.parent_list

    def parents(self, node, liste_parents, seen=set()):
        '''
        This function determines parent nodes for all nodes in our graph.
        This parent-children system will be later used in order to find a path between nodes quickly.
        This function relies on a DFS, so it is recursive.

        Parameters:
        -----------
            -input_graph : GraphType
                The graph which we consider. It is necessary for this graph to have been processed by the Kruskal algorithm.
            -node:  NodeType
                The node which we consider ; its neighbours become its children nodes, except for its parent.
            -liste_parents: List
                The list in which we store the parent-children system.
            -seen: Set
                The set of nodes which were already seen. It allows the DFS not to run twice on the same node.

        Output:
        -------
            -liste_parents:
                The list which represents the parent-children system.
        '''
        if node not in seen:
            #If the edge is not in the set, we add it and depth search from it.
            seen.add(node)
            list_of_neighbours = self.list_of_neighbours
            for neighbour in list_of_neighbours[node-1]:
                if neighbour not in seen:
                    liste_parents[neighbour-1] =  node #The node's parent is the node from which we come.
                    self.parents(neighbour,liste_parents,seen)
        return liste_parents

    def find_path_with_kruskal(self, origin, destination, power=-1):
        '''
        This function aims to find a path in a minimum spanning tree.
        The input_graph must have been processed by the Kruskal function.
        It relies on self.parents, that uses on a DFS.
        The complexity of this function is therefore given by build_parents's complexity(O(V))

        Parameters:
        -----------
            -input_graph : GraphType
                A graph object, which is a minimum spanning tree.
            -origin:    NodeType
                The starting node for the path.
            -destination :  NodeType
                The destination node for the path
            -power : integer(optionnal)
                The power which can restrict the edges for the algorithm to consider.
        Output:
        -------
            -power : integer
                The minimum power necessary to travel from the two nodes.
            -path: list
                The path from the two parameters nodes.
        '''
        #This function aims to find a quick path between two nodes, using the minimum spanning tree.
        list_of_parents = self.parent_list
        ancestors = set()
        #We build the list of all ancestors of the start node
        current_node = origin
        while list_of_parents[current_node-1] != current_node:
            ancestors.add(current_node)
            current_node = list_of_parents[current_node-1]
        ancestors.add(current_node)
        #To find the path, we find the lowest common ancestor of the two nodes.
        lca = destination
        while lca not in ancestors:
            lca = list_of_parents[lca-1]

        #The path is simple : starting node -> lca -> ending node
        ascending_path  = []
        descending_path = []
        current_node = origin
        while current_node != lca:
            ascending_path.append(current_node)
            current_node = list_of_parents[current_node-1]
        ascending_path.append(lca)

        current_node = destination
        while current_node != lca:
            descending_path.append(current_node)
            current_node = list_of_parents[current_node-1]
        #Now the path regardless of power is found.
        #To find the power, we collect all powers in that path, and identify the minimum
        path = ascending_path + descending_path[::-1]
        power = 0
        for index in range(len(path)-1):
            origin, destination = path[index], path[index+1]
            destination_index = self.list_of_neighbours[origin-1].index(destination)
            current_power = self.graph[origin][destination_index][1]
            if current_power > power:
                power = current_power
        return (power, path)

class Union_Find():
    '''
    This class is used later to implement the Kruskal's algorithm.
    It allows to create a Union-Find structure.
    '''
    def __init__(self):
        self.rank = None
        self.parent = None

    def make_set(self):
        self.parent = self
        self.rank = 0
    
    def find(self):
        node = self
        while node != node.parent:
            node = node.parent
        return node
    
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
    total_nodes = int(line_1[0])
    nb_edges = int(line_1[1].strip('\n'))
    new_graph = Graph([node for node in range(1,total_nodes+1)])
    new_graph.nb_edges = nb_edges
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
    new_graph.list_of_neighbours = [list(zip(*new_graph.graph[node]))[0] if new_graph.graph[node]!=[] else () for node in new_graph.nodes]
    file.close()
    return new_graph

def graph_into_pdf(filename):
    '''
    This function aims to draw a graph from a file into a pdf, using graphviz module.

    Parameters:
    -----------
        -filename : A compatible file to draw the graph.
    
    Output:
    -------
        -pdf : The pdf representing the graph.
    '''
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

def route_min_power(file):
    '''
    This function builds a graph, and determines the min_power for all routes proposed.
    Parameters:
    -----------
        -file : integer
            The integer representing the graph(eg : network.1 // routes.1)
    
    Output:
    -------
        -output : .in file
            The file containing all of the minimum power for all routes proposed.
    '''
    f = open(f'input/routes.{file}.in', 'r')
    g = graph_from_file(f'input/network.{file}.in')
    h = g.kruskal()
    h.build_parents()
    output = open(f'output/routes.{file}.out','w')
    output.write(f.readline())
    for line in f:#We read all lines to find the path
        list_line = line.split(' ')
        origin = int(list_line[0])
        destination = int(list_line[1])
        utility = int(list_line[2])
        min_power = h.find_path_with_kruskal(origin,destination)[0]#We determine min_power for this path
        output.write(str(min_power) + ' ' + str(utility))#We write in in our output file.
        output.write('\n')
    output.close()

def extract_values(file):
    f = open(f'output/routes.{file}.out', 'r')
    nb_trajets = int(f.readline())
    utility = np.zeros(nb_trajets)
    min_power = np.zeros(nb_trajets)
    for index, line in enumerate(f):
        current_utility = int(line.split(' ')[1])
        current_power = int(line.split(' ')[0])
        utility[index] = current_utility
        min_power[index] = current_power
    #The utility and min_power array of our routes are now properly initialized
    f.close()
    f = open(f'input/trucks.{file}.in', 'r')
    nb_trucks = int(f.readline())
    trucks = np.zeros((nb_trucks, 3))
    for index, line in enumerate(f):
        list_line = line.split(' ')
        power = int(list_line[0])
        cost = int(list_line[1])
        trucks[index][0] = index
        trucks[index][1] = cost
        trucks[index][2] = power
    f.close()
    #For each truck, we now have stored their costs in trucks-costs, and their powers in trucks_power
    minimal_cost = np.zeros((nb_trajets,2))
    for i in range(0,nb_trajets):#We check on all journeys 
        min_cost = 0
        min_truck = 0
        for j in range(0, nb_trucks):#We check on all trucks
            #WARNING : this approach relies on the fact that the trucks.in file is sorted by cost
            truck_cost = trucks[j][1]
            truck_power = trucks[j][2]
            if truck_power >= min_power[i]:
                min_truck = j
                min_cost = truck_cost
                break
        minimal_cost[i][0] = min_cost
        minimal_cost[i][1] = min_truck
    #Ok for optimal cost array
    return nb_trajets, utility, minimal_cost

def greedy_approach(nb_trajets, utility, minimal_cost, budget):
    utility_cost = np.zeros((nb_trajets, 2))
    for i in range(nb_trajets):
        utility_cost[i][0] = 0
        if utility[i] != 0:
            utility_cost[i][0] = utility[i]/minimal_cost[i][0]
        utility_cost[i][1] = i
    #We now have an array that stores the utility/cost value for each route
    sorted_utility = utility_cost[utility_cost[:,0].argsort()]

    output = np.zeros((nb_trajets, 4))
    for i in range(nb_trajets):
        rank = 139-i
        journey = int(sorted_utility[rank][1])
        if minimal_cost[journey][0] <= budget:#We have enough money to perform the route
            output[journey][0] = 1#1 if done, 0 if not
            output[journey][1] = minimal_cost[journey][0]#The cost of this route
            output[journey][2] = minimal_cost[journey][1]#The truck performing it
            output[journey][3] = utility[journey]#The utility of the route
        budget = budget - minimal_cost[journey][0]
    print(f'Utilité totale(méthode greedy) : {sum(output[i][3] for i in range(nb_trajets))}')

def dynamic_programming(nb_trajets, utility, minimal_cost, budget):
    W = budget
    weights = minimal_cost[:,0]
    value = utility
    new_W = int(W/50000)
    #Now we have two arrays : weights and values.
    M_matrix = np.empty([nb_trajets+1, new_W+1])
    for w in range(new_W+1):
        M_matrix[0, w] = 0
    for i in range(nb_trajets+1):
        M_matrix[i, 0] = 0
    #Now first column and first row are full of zeros
    for i in range(1, nb_trajets+1):
        for w in range(1, new_W+1):
            if weights[i-1] <= w*50000:#WARNING : this relies on the price difference between trucks
                M_matrix[i, w] = max(M_matrix[i-1, w-int(weights[i-1]//50000)] + int(value[i-1]), M_matrix[i-1, w])
            else:
                M_matrix[i,w] = M_matrix[i-1,w]
    print(f'Utilité totale(méthode knapsack) : {M_matrix[nb_trajets, new_W]}')
    

