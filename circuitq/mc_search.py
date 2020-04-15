import circuitq as cq

import numpy as np
import networkx as nx
import os
import copy
import pickle

# =============================================================================
# Directory handling
# =============================================================================
"""
Function that creates directories for data and figures if they don't exist
and deletes files within those directories.
Parameters
----------
file_name: str
    Name of directory to store data and figures

Returns
----------
absolute data_path: str
absolute figures_path: str
"""
def create_directories(file_name):
    main_dir = os.path.abspath('..')
    data_path ='data/mc_search_' + str(file_name) + '/'
    figures_path = 'figures/mc_search_' + str(file_name) + '/'
    for path in [data_path,figures_path]:
        file_dir = os.path.join(main_dir, path)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        else:
            for file in os.listdir(file_dir):
                file_path = os.path.join(file_dir, file)
                os.remove(file_path)

    return (os.path.join(main_dir, data_path),
           os.path.join(main_dir, figures_path) )

# =============================================================================
# Search run
# =============================================================================
"""
Optimization algorithm using the Metropolis-Hastings algorithm to search
for circuits

Parameters
----------
n_dim: int
    Dimension of the Hilbert space for 1-dimensional problems. The dimension gets
    scaled down if the dimensionality increases.
circuit_steps: int
    Number of iteration steps to vary the circuit (outer loop).
parameter_steps: int
    Number of iteration steps to vary the parameters (inner loop).
my_random: random.Random() instance
    The random instance for reproducability. Set a seed with random.seed(int) before.
file_name: str
    This name will be added to the file and figure directories.
tempreature: float
    Temperatur of the Metropolis-Hastings algorithm

Returns
----------
winner instance: numpy array
    Lists the value of the cost function, the paramter values and the graph of the
    optimized instance.
"""
def mc_search(n_dim, circuit_steps, parameter_steps, my_random, file_name, temperature):
    abs_data_path, abs_figure_path = create_directories(file_name)
    circuit, h_num, graph = initialize_transmon(n_dim)
    accepted_list = []
    cost_ct = get_cost(circuit, graph)
    with open(os.path.join(abs_data_path, 'search_data.pickle')
                ,'wb') as data_file:
        for c_s in range(circuit_steps):
            new_graph = do_action(graph, my_random)
            pickle.dump({"graph" + str(c_s) : new_graph} , data_file)
            new_circuit = cq.CircuitQ(new_graph)
            dimension = int(n_dim/(len(new_circuit.nodes_wo_ground)**2))
            new_circuit.get_numerical_hamiltonian(dimension)
            parameter_values = new_circuit.parameter_values
            new_circuit.get_eigensystem()
            new_cost_ct = get_cost(new_circuit, new_graph)
            cost_par = new_cost_ct
            cost_list = []
            values_list = []
            print("Initial parameters: " + str(parameter_values))
            for p_s in range(parameter_steps):
                varied_parameter_values = vary_parameter(parameter_values, my_random)
                pickle.dump({"parameter{}.{}".format(c_s,p_s): varied_parameter_values},
                            data_file)
                varied_h_num = new_circuit.get_numerical_hamiltonian(dimension,
                                                                     parameter_values=varied_parameter_values)
                new_circuit.get_eigensystem()
                new_cost_par = get_cost(new_circuit, new_graph)
                accept = accept_refuse(new_cost_par, cost_par, temperature, my_random)
                if accept:
                    parameter_values = varied_parameter_values
                    cost_list.append(new_cost_par)
                    values_list.append(varied_parameter_values)
                    cost_par = new_cost_par

            if accept_refuse(cost_par, cost_ct, temperature, my_random):
                print("\n graph accepted\n")
                graph = new_graph
                cost_ct = new_cost_ct
                cq.visualize_circuit_general(new_graph, abs_figure_path + str(c_s) + '_accepted_circuit' )
                if len(cost_list) > 0:
                    accepted_list.append([min(cost_list),values_list[cost_list.index(min((cost_list)))], new_graph])
            else:
                print("\n graph refused\n")
                cq.visualize_circuit_general(new_graph, abs_figure_path + str(c_s) + '_refused_circuit')
    print(accepted_list)
    accepted_list = np.array(accepted_list)
    lowest_cost = min(accepted_list[:,0])
    lowest_cost_pos = accepted_list[:,0].tolist().index(lowest_cost)
    winner_instance = accepted_list[lowest_cost_pos,:]

    return winner_instance

# =============================================================================
# Initialize Transmon
# =============================================================================
def initialize_transmon(n_dim):
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, element='C')
    graph.add_edge(0, 1, element='J')
    circuit = cq.CircuitQ(graph)
    h_num = circuit.get_numerical_hamiltonian(n_dim)
    circuit.get_eigensystem()

    return circuit, h_num, graph

# =============================================================================
# Get value of the cost function for given instance
# =============================================================================
def get_cost(circuit, graph):
    anharmonicity = circuit.get_spectrum_anharmonicity() *10**0
    nbr_edges = len(graph.edges)
    cost = 1-anharmonicity + nbr_edges
    print("Anharmonicity = " + str(anharmonicity))
    print("Number of edges = " + str(nbr_edges))
    return cost

# =============================================================================
# Possible graph modifications
# =============================================================================

# Add element in between two existing nodes
# =============================================================================
def add_edge(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    u = my_random.choice(nodes)
    nodes.remove(u)
    v = my_random.choice(nodes)
    rdm_element = my_random.choice(elements)
    graph.add_edge(u, v, element = rdm_element)
    return graph

# Insert element in between two existing elements
# =============================================================================
def insert_edge(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    rdm_node = my_random.choice(nodes)
    rdm_edge = my_random.choice(list(graph.edges(rdm_node, keys=True, data=True)))
    graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
    new_node = max(nodes) + 1
    rdm_case_nbr = my_random.uniform(0, 1)
    rdm_element = my_random.choice(elements)
    if rdm_case_nbr < 0.5:
        graph.add_edge(rdm_edge[0], new_node, **rdm_edge[3])
        graph.add_edge(new_node, rdm_edge[1], element = rdm_element)
    elif 0.5 <= rdm_case_nbr <= 1:
        graph.add_edge(new_node, rdm_edge[1], **rdm_edge[3])
        graph.add_edge(rdm_edge[0], new_node, element = rdm_element)
    return graph

# Add new loop to existing node
# =============================================================================
def add_loop(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    rdm_node = my_random.choice(nodes)
    new_node = max(nodes) + 1
    rdm_element = my_random.choice(elements)
    graph.add_edge(rdm_node, new_node, element=rdm_element)
    rdm_element_new = my_random.choice(elements)
    while rdm_element_new == rdm_element:
        rdm_element_new = my_random.choice(elements)
    graph.add_edge(rdm_node, new_node, element=rdm_element_new)
    return graph

# Remove edge
# =============================================================================
def remove_edge(graph, my_random, no_deletion = 0):
    graph = copy.deepcopy(graph)
    edges = list(graph.edges.data(keys=True))
    rdm_edge = my_random.choice(edges)
    if len(edges) == 2:
        print("Only two edges. No deletion!")
        return do_action(graph, my_random)
    ngb1 = list(graph.neighbors(rdm_edge[0]))
    nbr_ngb1 = len(ngb1)
    ngb2 = list(graph.neighbors(rdm_edge[1]))
    nbr_ngb2 = len(ngb2)
    if nbr_ngb1 > 1 and nbr_ngb2 > 1:
        graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
        if not graph.has_edge(rdm_edge[0], rdm_edge[1]):
            graph = nx.contracted_nodes(graph, rdm_edge[0], rdm_edge[1])
        return graph
    # If the edge has one boundary node:
    else:
        nbr_edges = graph.number_of_edges(rdm_edge[0], rdm_edge[1])
        if nbr_edges > 2:
            graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
            return graph
        # Delete loop if it's connected to the rest of the graph with at
        # least two edges
        elif nbr_edges == 2:
            if nbr_ngb1 > 1:
                connected_node = rdm_edge[0]
                same_edge_node = rdm_edge[1]
            elif nbr_ngb2 > 1:
                connected_node = rdm_edge[1]
                same_edge_node = rdm_edge[0]
            else:
                raise Exception("Isolated subgraph detected!")
            connected_ngb = list(graph.neighbors(connected_node))
            connected_ngb.remove(same_edge_node)
            if len(connected_ngb)>1:
                delete = True
            else:
                nbr_connect_edges = graph.number_of_edges(connected_node, connected_ngb[0])
                if nbr_connect_edges >1 :
                    for edge in edges:
                        if ((edge[0] == rdm_edge[0] and edge[1] == rdm_edge[1])
                                or (edge[0] == rdm_edge[1] and edge[1] == rdm_edge[0])):
                            graph.remove_edge(edge[0], edge[1], key=edge[2])
                        isolates = list(nx.isolates(graph))
                        for node in isolates:
                            graph.remove_node(node)
                    return graph
                else:
                    print("Loop could not be deleted (weakly connected)")
                    no_deletion_new = no_deletion + 1
                    if no_deletion_new > 20:
                        raise Exception("Might be hard or impossible to delete further elements.")
                    return remove_edge(graph, my_random, no_deletion_new)
        else:
            raise Exception("Only one edge connected to boundary node")

# =============================================================================
# Perform a graph modification
# =============================================================================
def do_action(graph, my_random):
    elements = ['C','L','J']
    actions = ['add_edge', 'insert_edge', 'add_loop', 'remove_edge']
    rdm_action = my_random.choice(actions)
    print("\nAction = " + rdm_action)
    if rdm_action == 'add_edge':
        return add_edge(graph, my_random, elements)
    elif rdm_action == 'insert_edge':
        return insert_edge(graph, my_random, elements)
    elif rdm_action == 'add_loop':
        return add_loop(graph, my_random, elements)
    elif rdm_action == 'remove_edge':
        return remove_edge(graph, my_random)
    else:
        raise Exception("Action not recognized.")

# =============================================================================
# Vary parameter of circuit elements
# =============================================================================
def vary_parameter(parameter_values, my_random):
    parameter_values = copy.deepcopy(parameter_values)
    parameter = my_random.choice(parameter_values)
    parameter_idx = parameter_values.index(parameter)
    print("Changed parameter " + str(parameter))
    if parameter == 0:
        parameter = 1
    else:
        rdm_factor = my_random.choice([2/3, 4/3])
        parameter = rdm_factor*parameter
    parameter_values[parameter_idx] = parameter
    print("to " + str(parameter))
    return parameter_values

# =============================================================================
# Accept or Refuse
# =============================================================================
def accept_refuse(new_cost, old_cost, temperature, my_random):
    exp_arg = float((old_cost - new_cost) / temperature)
    p_accept = min([1, np.exp(exp_arg)])
    print("Acceptance probability: " + str(p_accept))
    random_nbr = my_random.uniform(0, 1)
    if random_nbr <= p_accept:
        print("accepting parameter change")
        return True
    print("refusing parameter change")
    return False

